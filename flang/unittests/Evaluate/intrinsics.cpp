#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/target.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/provenance.h"
#include "flang/Testing/testing.h"
#include "llvm/Support/raw_ostream.h"
#include <initializer_list>
#include <map>
#include <string>

namespace Fortran::evaluate {

class CookedStrings {
public:
  CookedStrings() {}
  explicit CookedStrings(const std::initializer_list<std::string> &ss) {
    for (const auto &s : ss) {
      Save(s);
    }
    Marshal();
  }
  void Save(const std::string &s) {
    offsets_[s] = cooked_.Put(s);
    cooked_.PutProvenance(allSources_.AddCompilerInsertion(s));
  }
  void Marshal() { cooked_.Marshal(allCookedSources_); }
  parser::CharBlock operator()(const std::string &s) {
    return {cooked_.AsCharBlock().begin() + offsets_[s], s.size()};
  }
  parser::ContextualMessages Messages(parser::Messages &buffer) {
    return parser::ContextualMessages{cooked_.AsCharBlock(), &buffer};
  }
  void Emit(llvm::raw_ostream &o, const parser::Messages &messages) {
    messages.Emit(o, allCookedSources_);
  }

private:
  parser::AllSources allSources_;
  parser::AllCookedSources allCookedSources_{allSources_};
  parser::CookedSource &cooked_{allCookedSources_.NewCookedSource()};
  std::map<std::string, std::size_t> offsets_;
};

// Since the migration to runtime kinds, Fortran's Type<CAT> no longer carries
// its kind as a template parameter, so the kind can no longer be recovered from
// a (kindless) scalar value alone.  This compile-time tag bundles the kindless
// Fortran Type<CAT> with the kind that the test wants, so the existing
// IntN/RealN/... aliases keep working: TagN::GetType() yields the runtime
// DynamicType, Scalar<TagN> resolves to the runtime-kind Scalar, and
// Const<TagN> stamps the kind onto the constant's result type.
template <common::TypeCategory CAT, int KIND> struct TestTypeKind {
  static constexpr common::TypeCategory category{CAT};
  static constexpr int kind{KIND};
  using FortranType = Type<CAT>;
  using Scalar = Fortran::evaluate::Scalar<FortranType>;
  static constexpr DynamicType GetType() { return DynamicType{CAT, KIND}; }
};

// With the collapsed-kind scalars, TypeOf<> can no longer recover the kind of a
// scalar (all kinds of a category share one Scalar type), so a TestTypeKind tag
// may be supplied to stamp the desired kind onto the constant:
// Const<Int4>(...).
template <typename T = void, typename A> auto Const(A &&x) {
  if constexpr (std::is_void_v<T>) {
    return Constant<TypeOf<A>>{std::move(x)};
  } else {
    using FT = typename T::FortranType;
    constexpr int kind{T::kind};
    // A default-constructed (monostate) numeric scalar cannot report its kind,
    // which formatting and folding require, so materialize a zero value in the
    // requested runtime kind.
    if constexpr (FT::category == TypeCategory::Integer ||
        FT::category == TypeCategory::Unsigned) {
      return Constant<FT>{Scalar<FT>{x.ToInt64(), kind}, FT{kind}};
    } else if constexpr (FT::category == TypeCategory::Real ||
        FT::category == TypeCategory::Complex) {
      return Constant<FT>{Scalar<FT>::Zero(kind), FT{kind}};
    } else {
      return Constant<FT>{std::move(x), FT{kind}};
    }
  }
}

template <typename A> struct NamedArg {
  std::string keyword;
  A value;
};

template <typename A> static NamedArg<A> Named(std::string kw, A &&x) {
  return {kw, std::move(x)};
}

struct TestCall {
  TestCall(const common::IntrinsicTypeDefaultKinds &d,
      const IntrinsicProcTable &t, std::string n)
      : defaults{d}, table{t}, name{n} {}
  template <typename A> TestCall &Push(A &&x) {
    args.emplace_back(AsGenericExpr(std::move(x)));
    keywords.push_back("");
    return *this;
  }
  template <typename A> TestCall &Push(NamedArg<A> &&x) {
    args.emplace_back(AsGenericExpr(std::move(x.value)));
    keywords.push_back(x.keyword);
    strings.Save(x.keyword);
    return *this;
  }
  template <typename A, typename... As> TestCall &Push(A &&x, As &&...xs) {
    Push(std::move(x));
    return Push(std::move(xs)...);
  }
  void Marshal() {
    strings.Save(name);
    strings.Marshal();
    std::size_t j{0};
    for (auto &kw : keywords) {
      if (!kw.empty()) {
        args[j]->set_keyword(strings(kw));
      }
      ++j;
    }
  }
  void DoCall(std::optional<DynamicType> resultType = std::nullopt,
      int rank = 0, bool isElemental = false) {
    Marshal();
    parser::CharBlock fName{strings(name)};
    llvm::outs() << "function: " << fName.ToString();
    char sep{'('};
    for (const auto &a : args) {
      llvm::outs() << sep;
      sep = ',';
      a->AsFortran(llvm::outs());
    }
    if (sep == '(') {
      llvm::outs() << '(';
    }
    llvm::outs() << ')' << '\n';
    llvm::outs().flush();
    CallCharacteristics call{fName.ToString()};
    auto messages{strings.Messages(buffer)};
    TargetCharacteristics targetCharacteristics;
    common::LanguageFeatureControl languageFeatures;
    FoldingContext context{messages, defaults, table, targetCharacteristics,
        languageFeatures, tempNames};
    std::optional<SpecificCall> si{table.Probe(call, args, context)};
    if (resultType.has_value()) {
      TEST(si.has_value());
      TEST(messages.messages() && !messages.messages()->AnyFatalError());
      if (si) {
        const auto &proc{si->specificIntrinsic.characteristics.value()};
        const auto &fr{proc.functionResult};
        TEST(fr.has_value());
        if (fr) {
          const auto *ts{fr->GetTypeAndShape()};
          TEST(ts != nullptr);
          if (ts) {
            TEST(*resultType == ts->type());
            MATCH(rank, ts->Rank());
          }
        }
        MATCH(isElemental,
            proc.attrs.test(characteristics::Procedure::Attr::Elemental));
      }
    } else {
      TEST(!si.has_value());
      TEST((messages.messages() && messages.messages()->AnyFatalError()) ||
          name == "bad");
    }
    strings.Emit(llvm::outs(), buffer);
  }

  const common::IntrinsicTypeDefaultKinds &defaults;
  const IntrinsicProcTable &table;
  CookedStrings strings;
  parser::Messages buffer;
  ActualArguments args;
  std::string name;
  std::vector<std::string> keywords;
  std::set<std::string> tempNames;
};

void TestIntrinsics() {
  common::IntrinsicTypeDefaultKinds defaults;
  MATCH(4, defaults.GetDefaultKind(TypeCategory::Integer));
  MATCH(4, defaults.GetDefaultKind(TypeCategory::Real));
  IntrinsicProcTable table{IntrinsicProcTable::Configure(defaults)};
  table.Dump(llvm::outs());

  using Int1 = TestTypeKind<TypeCategory::Integer, 1>;
  using Int4 = TestTypeKind<TypeCategory::Integer, 4>;
  using Int8 = TestTypeKind<TypeCategory::Integer, 8>;
  using Real4 = TestTypeKind<TypeCategory::Real, 4>;
  using Real8 = TestTypeKind<TypeCategory::Real, 8>;
  using Complex4 = TestTypeKind<TypeCategory::Complex, 4>;
  using Complex8 = TestTypeKind<TypeCategory::Complex, 8>;
  using Char = TestTypeKind<TypeCategory::Character, 1>;
  using Log4 = TestTypeKind<TypeCategory::Logical, 4>;

  TestCall{defaults, table, "bad"}
      .Push(Const<Int4>(Scalar<Int4>{}))
      .DoCall(); // bad intrinsic name
  TestCall{defaults, table, "abs"}
      .Push(Named("a", Const<Int4>(Scalar<Int4>{})))
      .DoCall(Int4::GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const<Int4>(Scalar<Int4>{}))
      .DoCall(Int4::GetType());
  TestCall{defaults, table, "abs"}
      .Push(Named("bad", Const<Int4>(Scalar<Int4>{})))
      .DoCall(); // bad keyword
  TestCall{defaults, table, "abs"}.DoCall(); // insufficient args
  TestCall{defaults, table, "abs"}
      .Push(Const<Int4>(Scalar<Int4>{}))
      .Push(Const<Int4>(Scalar<Int4>{}))
      .DoCall(); // too many args
  TestCall{defaults, table, "abs"}
      .Push(Const<Int4>(Scalar<Int4>{}))
      .Push(Named("a", Const<Int4>(Scalar<Int4>{})))
      .DoCall();
  TestCall{defaults, table, "abs"}
      .Push(Named("a", Const<Int4>(Scalar<Int4>{})))
      .Push(Const<Int4>(Scalar<Int4>{}))
      .DoCall();
  TestCall{defaults, table, "abs"}
      .Push(Const<Int1>(Scalar<Int1>{}))
      .DoCall(Int1::GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const<Int4>(Scalar<Int4>{}))
      .DoCall(Int4::GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const<Int8>(Scalar<Int8>{}))
      .DoCall(Int8::GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const<Real4>(Scalar<Real4>{}))
      .DoCall(Real4::GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const<Real8>(Scalar<Real8>{}))
      .DoCall(Real8::GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const<Complex4>(Scalar<Complex4>{}))
      .DoCall(Real4::GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const<Complex8>(Scalar<Complex8>{}))
      .DoCall(Real8::GetType());
  TestCall{defaults, table, "abs"}
      .Push(Const(Scalar<Char>::Zero(Char::kind)))
      .DoCall();
  TestCall{defaults, table, "abs"}.Push(Const(Scalar<Log4>{})).DoCall();

  // "Ext" in names for calls allowed as extensions
  TestCall maxCallR{defaults, table, "max"}, maxCallI{defaults, table, "min"},
      max0Call{defaults, table, "max0"}, max1Call{defaults, table, "max1"},
      amin0Call{defaults, table, "amin0"}, amin1Call{defaults, table, "amin1"},
      max0ExtCall{defaults, table, "max0"},
      amin1ExtCall{defaults, table, "amin1"};
  for (int j{0}; j < 10; ++j) {
    maxCallR.Push(Const<Real4>(Scalar<Real4>{}));
    maxCallI.Push(Const<Int4>(Scalar<Int4>{}));
    max0Call.Push(Const<Int4>(Scalar<Int4>{}));
    max0ExtCall.Push(Const<Real4>(Scalar<Real4>{}));
    max1Call.Push(Const<Real4>(Scalar<Real4>{}));
    amin0Call.Push(Const<Int4>(Scalar<Int4>{}));
    amin1ExtCall.Push(Const<Int4>(Scalar<Int4>{}));
    amin1Call.Push(Const<Real4>(Scalar<Real4>{}));
  }
  maxCallR.DoCall(Real4::GetType());
  maxCallI.DoCall(Int4::GetType());
  max0Call.DoCall(Int4::GetType());
  max0ExtCall.DoCall(Int4::GetType());
  max1Call.DoCall(Int4::GetType());
  amin0Call.DoCall(Real4::GetType());
  amin1Call.DoCall(Real4::GetType());
  amin1ExtCall.DoCall(Real4::GetType());

  TestCall{defaults, table, "conjg"}
      .Push(Const<Complex4>(Scalar<Complex4>{}))
      .DoCall(Complex4::GetType());
  TestCall{defaults, table, "conjg"}
      .Push(Const<Complex8>(Scalar<Complex8>{}))
      .DoCall(Complex8::GetType());
  TestCall{defaults, table, "dconjg"}
      .Push(Const<Complex8>(Scalar<Complex8>{}))
      .DoCall(Complex8::GetType());

  TestCall{defaults, table, "float"}
      .Push(Const<Real4>(Scalar<Real4>{}))
      .DoCall();
  TestCall{defaults, table, "float"}
      .Push(Const<Int4>(Scalar<Int4>{}))
      .DoCall(Real4::GetType());
  TestCall{defaults, table, "idint"}.Push(Const<Int4>(Scalar<Int4>{})).DoCall();
  TestCall{defaults, table, "idint"}
      .Push(Const<Real8>(Scalar<Real8>{}))
      .DoCall(Int4::GetType());

  // Allowed as extensions
  TestCall{defaults, table, "float"}
      .Push(Const<Int8>(Scalar<Int8>{}))
      .DoCall(Real4::GetType());
  TestCall{defaults, table, "idint"}
      .Push(Const<Real4>(Scalar<Real4>{}))
      .DoCall(Int4::GetType());

  TestCall{defaults, table, "num_images"}.DoCall(Int4::GetType());
  TestCall{defaults, table, "num_images"}
      .Push(Const<Int1>(Scalar<Int1>{}))
      .DoCall(Int4::GetType());
  TestCall{defaults, table, "num_images"}
      .Push(Const<Int4>(Scalar<Int4>{}))
      .DoCall(Int4::GetType());
  TestCall{defaults, table, "num_images"}
      .Push(Const<Int8>(Scalar<Int8>{}))
      .DoCall(Int4::GetType());
  TestCall{defaults, table, "num_images"}
      .Push(Named("team_number", Const<Int4>(Scalar<Int4>{})))
      .DoCall(Int4::GetType());
  TestCall{defaults, table, "num_images"}
      .Push(Const<Int4>(Scalar<Int4>{}))
      .Push(Const<Int4>(Scalar<Int4>{}))
      .DoCall(); // too many args
  TestCall{defaults, table, "num_images"}
      .Push(Named("bad", Const<Int4>(Scalar<Int4>{})))
      .DoCall(); // bad keyword
  TestCall{defaults, table, "num_images"}
      .Push(Const(Scalar<Char>::Zero(Char::kind)))
      .DoCall(); // bad type
  TestCall{defaults, table, "num_images"}
      .Push(Const(Scalar<Log4>{}))
      .DoCall(); // bad type
  TestCall{defaults, table, "num_images"}
      .Push(Const<Complex8>(Scalar<Complex8>{}))
      .DoCall(); // bad type
  TestCall{defaults, table, "num_images"}
      .Push(Const<Real4>(Scalar<Real4>{}))
      .DoCall(); // bad type

  // This test temporarily removed because it requires access to
  // the ISO_FORTRAN_ENV intrinsic module. This module should to
  // be loaded (somehow) and the following test reinstated.
  // TestCall{defaults, table, "team_number"}.DoCall(Int4::GetType());

  TestCall{defaults, table, "team_number"}
      .Push(Const<Int4>(Scalar<Int4>{}))
      .Push(Const<Int4>(Scalar<Int4>{}))
      .DoCall(); // too many args
  TestCall{defaults, table, "team_number"}
      .Push(Named("bad", Const<Int4>(Scalar<Int4>{})))
      .DoCall(); // bad keyword
  TestCall{defaults, table, "team_number"}
      .Push(Const<Int4>(Scalar<Int4>{}))
      .DoCall(); // bad type
  TestCall{defaults, table, "team_number"}
      .Push(Const(Scalar<Char>::Zero(Char::kind)))
      .DoCall(); // bad type
  TestCall{defaults, table, "team_number"}
      .Push(Const(Scalar<Log4>{}))
      .DoCall(); // bad type
  TestCall{defaults, table, "team_number"}
      .Push(Const<Complex8>(Scalar<Complex8>{}))
      .DoCall(); // bad type
  TestCall{defaults, table, "team_number"}
      .Push(Const<Real4>(Scalar<Real4>{}))
      .DoCall(); // bad type

  // TODO: test other intrinsics

  // Test unrestricted specific to generic name mapping (table 16.2).
  TEST(table.GetGenericIntrinsicName("alog") == "log");
  TEST(table.GetGenericIntrinsicName("alog10") == "log10");
  TEST(table.GetGenericIntrinsicName("amod") == "mod");
  TEST(table.GetGenericIntrinsicName("cabs") == "abs");
  TEST(table.GetGenericIntrinsicName("ccos") == "cos");
  TEST(table.GetGenericIntrinsicName("cexp") == "exp");
  TEST(table.GetGenericIntrinsicName("clog") == "log");
  TEST(table.GetGenericIntrinsicName("csin") == "sin");
  TEST(table.GetGenericIntrinsicName("csqrt") == "sqrt");
  TEST(table.GetGenericIntrinsicName("dabs") == "abs");
  TEST(table.GetGenericIntrinsicName("dacos") == "acos");
  TEST(table.GetGenericIntrinsicName("dasin") == "asin");
  TEST(table.GetGenericIntrinsicName("datan") == "atan");
  TEST(table.GetGenericIntrinsicName("datan2") == "atan2");
  TEST(table.GetGenericIntrinsicName("dcos") == "cos");
  TEST(table.GetGenericIntrinsicName("dcosh") == "cosh");
  TEST(table.GetGenericIntrinsicName("ddim") == "dim");
  TEST(table.GetGenericIntrinsicName("derf") == "erf");
  TEST(table.GetGenericIntrinsicName("dexp") == "exp");
  TEST(table.GetGenericIntrinsicName("dint") == "aint");
  TEST(table.GetGenericIntrinsicName("dlog") == "log");
  TEST(table.GetGenericIntrinsicName("dlog10") == "log10");
  TEST(table.GetGenericIntrinsicName("dmod") == "mod");
  TEST(table.GetGenericIntrinsicName("dnint") == "anint");
  TEST(table.GetGenericIntrinsicName("dsign") == "sign");
  TEST(table.GetGenericIntrinsicName("dsin") == "sin");
  TEST(table.GetGenericIntrinsicName("dsinh") == "sinh");
  TEST(table.GetGenericIntrinsicName("dsqrt") == "sqrt");
  TEST(table.GetGenericIntrinsicName("dtan") == "tan");
  TEST(table.GetGenericIntrinsicName("dtanh") == "tanh");
  TEST(table.GetGenericIntrinsicName("iabs") == "abs");
  TEST(table.GetGenericIntrinsicName("idim") == "dim");
  TEST(table.GetGenericIntrinsicName("idnint") == "nint");
  TEST(table.GetGenericIntrinsicName("isign") == "sign");
  // Test a case where specific and generic name are the same.
  TEST(table.GetGenericIntrinsicName("acos") == "acos");
}
} // namespace Fortran::evaluate

int main() {
  Fortran::evaluate::TestIntrinsics();
  return testing::Complete();
}
