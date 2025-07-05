#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_C2000_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_C2000_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
namespace targets {

class C2000TargetInfo : public TargetInfo {
  static const char *const GCCRegNames[];
  bool eabi = false;
  bool strict = false;
  bool opt = false;
  bool fpu64 = false;
  bool fpu32 = false;
  bool relaxed = false;
  bool tmu_support = false;
  bool cla_support = false;
  bool vcu_support = false;
  bool cla0 = false;
  bool cla1 = false;
  bool cla2 = false;
  bool vcu2 = false;
  bool vcrc = false;
  bool tmu1 = false;
  bool idiv0 = false;

public:
  C2000TargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    TLSSupported = false;
    BigEndian = false;
    BoolWidth = BoolAlign = 16;
    IntWidth = IntAlign = 16;
    LongLongWidth = 64;
    LongLongAlign = 32;
    FloatWidth = FloatAlign = 32;
    DoubleWidth = 64;
    DoubleAlign = 32;
    LongDoubleWidth = 64;
    LongDoubleAlign = 32;
    PointerWidth = 32;
    PointerAlign = 16;
    Char16Type = UnsignedShort;
    Char32Type = UnsignedLong;
    SizeType = UnsignedLong;
    PtrDiffType = SignedLong;
    WCharType = UnsignedLong;
    WIntType = UnsignedLong;
    IntMaxType = SignedLongLong;
    IntPtrType = SignedInt;
    resetDataLayout(
        "e-m:e-p:32:16-i16:16-i32:32-i64:32-f32:32-f64:32-a:8-n16:32-S32");
  }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override;

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  ArrayRef<Builtin::Info> getTargetBuiltins() const override { return {}; }

  bool hasFeature(StringRef Feature) const override;

  ArrayRef<const char *> getGCCRegNames() const override;

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    // Make these be recognized by llc (f.e., in clobber list)
    static const TargetInfo::GCCRegAlias GCCRegAliases[] = {
        {{"xt"}, "mr"}, {{"p"}, "pr"},    {{"dp"}, "dpp"}, {{"sp"}, "sp"},
        {{"pc"}, "pc"}, {{"rpc"}, "rpc"}, {{"st0"}, "sr"}, {{"ifr"}, "icr"},
    };
    return llvm::ArrayRef(GCCRegAliases);
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {

    return false;
  }

  std::string_view getClobbers() const override {
    // FIXME: Is this really right?
    return "";
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    // FIXME: implement
    return TargetInfo::CharPtrBuiltinVaList;
  }
};

} // namespace targets
} // namespace clang

#endif
