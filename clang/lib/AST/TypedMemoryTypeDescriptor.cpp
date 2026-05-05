#include "clang/AST/TypedMemoryTypeDescriptor.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/Support/SipHash.h"

namespace clang {

bool TypedMemoryTypeDescriptor::initialize(const QualType &QT,
                                           const TypedMemoryLayout &Layout) {
  Flags = TypedMemoryTypeFlags::None;

  initializeLayoutProperties(Layout);

  if (const CXXRecordDecl *RD = QT->getAsCXXRecordDecl()) {
    if (RD->isPolymorphic()) {
      Flags |= TypedMemoryTypeFlags::IsPolymorphic;
    }
  }

  return true;
}

bool TypedMemoryTypeDescriptor::initializeLayoutProperties(
    const TypedMemoryLayout &Layout) {
  LayoutProperties.clear();
  LayoutSemantics = TypedMemoryLayoutSemantics::None;

  auto &Fields = Layout.Fields;
  for (auto &F : Fields) {
    if (F.Width > 0) {
      auto S = getFieldSemantics(&F);
      LayoutSemantics |= S;
      LayoutProperties.push_back({F.Offset, F.Width, S});
    }
  }

  llvm::stable_sort(LayoutProperties,
                    [&](LayoutSemanticsSpan L, LayoutSemanticsSpan R) {
                      return L.Offset < R.Offset;
                    });

  initializeCoalescedLayoutProperties(Layout);
  return true;
}

void TypedMemoryTypeDescriptor::initializeCoalescedLayoutProperties(
    const TypedMemoryLayout &Layout) {
  CoalescedLayoutProperties.clear();
  if (LayoutProperties.empty() || Layout.Width == 0)
    return;

  SmallVector<TypedMemoryLayoutSemantics> Unfolded;
  Unfolded.resize(Layout.Width, TypedMemoryLayoutSemantics::None);

  for (auto &Cur : LayoutProperties) {
    for (size_t I = Cur.Offset; I < Cur.Offset + Cur.Width; I++) {
      if (Unfolded[I] != TypedMemoryLayoutSemantics::None &&
          Unfolded[I] != Cur.Semantics) {
        Flags |= TypedMemoryTypeFlags::HasMixedUnions;
      }
      Unfolded[I] |= Cur.Semantics;
    }
  }

  CoalescedLayoutProperties.push_back({0, 0, Unfolded.front()});
  for (size_t I = 0; I < Unfolded.size(); I++) {
    LayoutSemanticsSpan &Last = CoalescedLayoutProperties.back();
    if (Unfolded[I] == Last.Semantics) {
      Last.Width += 1;
    } else {
      CoalescedLayoutProperties.push_back({I, 1, Unfolded[I]});
    }
  }
}

TypedMemoryLayoutSemantics TypedMemoryTypeDescriptor::getFieldSemantics(
    const TypedMemoryLayoutField *F) const {
  // TODO: Process any field attribute here
  return getTypeSemantics(F->Type);
}

TypedMemoryLayoutSemantics
TypedMemoryTypeDescriptor::getTypeSemantics(const QualType &QType) const {

  QualType QT = QType;
  TypedMemoryLayoutSemantics S = TypedMemoryLayoutSemantics::None;

  // For _Atomic-qualified types, retrieve the value type, so that we compute
  // the encoding of the underlying semantics of the type.
  if (auto *AT = QT->getAs<AtomicType>())
    QT = AT->getValueType();

  QualType Pointee = QT->getPointeeType();
  if (!Pointee.isNull()) {
    if (Pointee->isIntegerType()) {
      // pointer-to-data
      S |= TypedMemoryLayoutSemantics::DataPointer;
    } else if (Pointee->isRecordType() || Pointee->isUnionType()) {
      // pointer-to-structure
      S |= TypedMemoryLayoutSemantics::StructPointer;
    } else {
      // anonymous-pointer
      S |= TypedMemoryLayoutSemantics::AnonymousPointer;
    }

    if ((QT.isConstQualified() || Pointee.isConstQualified()) ||
        QT.getLocalQualifiers().getPointerAuth().withoutKeyNone()) {
      // immutable-pointer
      S |= TypedMemoryLayoutSemantics::ImmutablePointer;
    }
  } else {
    // TODO: For now, we don't have other encodings.
    S |= TypedMemoryLayoutSemantics::GenericData;

    // There are magic pointer-like values that are ostensibly
    // opaque "data", that for now we will simply report as
    // GenericData | AnonymousPointer. If some of these special
    // values become sufficiently important we can add additional
    // diversification in future.
    if (QT->hasPointerRepresentation())
      S |= TypedMemoryLayoutSemantics::AnonymousPointer;
  }

  return S;
}

uint32_t TypedMemoryTypeDescriptor::computeTypeHash() const {
  std::string layoutString;
  llvm::raw_string_ostream layoutStream(layoutString);
  for (auto &S : CoalescedLayoutProperties) {
    layoutStream << "[" << S.Offset << "," << S.Width << ":"
                 << static_cast<uint16_t>(S.Semantics) << "]";
  }
  return llvm::getTypedMemoryDescriptorStableSipHash(layoutString) & 0xffffffff;
}

uint32_t TypedMemoryTypeDescriptor::computeTypeNameHash(const QualType &QT) {
  std::string TypeName = QT.getAsString();
  return llvm::getTypedMemoryDescriptorStableSipHash(TypeName) & 0xffffffff;
}

} // namespace clang
