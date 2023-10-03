#include "../clang-tidy/utils/DeclRefExprUtils.h"
#include "ClangTidyDiagnosticConsumer.h"
#include "ClangTidyTest.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {

namespace {
using namespace clang::ast_matchers;

class ConstReferenceDeclRefExprsTransform : public ClangTidyCheck {
public:
  ConstReferenceDeclRefExprsTransform(StringRef CheckName,
                                      ClangTidyContext *Context)
      : ClangTidyCheck(CheckName, Context) {}

  void registerMatchers(MatchFinder *Finder) override {
    Finder->addMatcher(varDecl(hasName("target")).bind("var"), this);
  }

  void check(const MatchFinder::MatchResult &Result) override {
    const auto *D = Result.Nodes.getNodeAs<VarDecl>("var");
    using utils::decl_ref_expr::constReferenceDeclRefExprs;
    const auto const_decrefexprs = constReferenceDeclRefExprs(
        *D, *cast<FunctionDecl>(D->getDeclContext())->getBody(),
        *Result.Context);

    for (const DeclRefExpr *const Expr : const_decrefexprs) {
      assert(Expr);
      diag(Expr->getBeginLoc(), "const usage")
          << FixItHint::CreateInsertion(Expr->getBeginLoc(), "/*const*/");
    }
  }
};
} // namespace

namespace test {

void RunTest(StringRef Snippet) {

  StringRef CommonCode = R"(
    struct ConstTag{};
    struct NonConstTag{};

    struct S {
      void constMethod() const;
      void nonConstMethod();

      void operator()(ConstTag) const;
      void operator()(NonConstTag);

      void operator[](int);
      void operator[](int) const;

      bool operator==(const S&) const;

      int int_member;
      int* ptr_member;

    };

    struct Derived : public S {

    };

    void useVal(S);
    void useRef(S&);
    void usePtr(S*);
    void usePtrPtr(S**);
    void usePtrConstPtr(S* const*);
    void useConstRef(const S&);
    void useConstPtr(const S*);
    void useConstPtrRef(const S*&);
    void useConstPtrPtr(const S**);
    void useConstPtrConstRef(const S* const&);
    void useConstPtrConstPtr(const S* const*);

    void useInt(int);
    void useIntRef(int&);
    void useIntConstRef(const int&);
    void useIntPtr(int*);
    void useIntConstPtr(const int*);

    )";

  std::string Code = (CommonCode + Snippet).str();

  llvm::SmallVector<StringRef, 1> Parts;
  StringRef(Code).split(Parts, "/*const*/");

  EXPECT_EQ(Code, runCheckOnCode<ConstReferenceDeclRefExprsTransform>(
                      join(Parts, "")));
}

TEST(ConstReferenceDeclRefExprsTest, ConstValueVar) {
  RunTest(R"(
    void f(const S target) {
      useVal(/*const*/target);
      useConstRef(/*const*/target);
      useConstPtr(&target);
      useConstPtrConstRef(&target);
      /*const*/target.constMethod();
      /*const*/target(ConstTag{});
      /*const*/target[42];
      useConstRef((/*const*/target));
      (/*const*/target).constMethod();
      (void)(/*const*/target == /*const*/target);
      (void)target;
      (void)&target;
      (void)*&target;
      S copy1 = /*const*/target;
      S copy2(/*const*/target);
      useInt(/*const*/target.int_member);
      useIntConstRef(/*const*/target.int_member);
      useIntPtr(target.ptr_member);
      useIntConstPtr(&target.int_member);
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, ConstRefVar) {
  RunTest(R"(
    void f(const S& target) {
      useVal(/*const*/target);
      useConstRef(/*const*/target);
      useConstPtr(&target);
      useConstPtrConstRef(&target);
      /*const*/target.constMethod();
      /*const*/target(ConstTag{});
      /*const*/target[42];
      useConstRef((/*const*/target));
      (/*const*/target).constMethod();
      (void)(/*const*/target == /*const*/target);
      (void)target;
      (void)&target;
      (void)*&target;
      S copy1 = /*const*/target;
      S copy2(/*const*/target);
      useInt(/*const*/target.int_member);
      useIntConstRef(/*const*/target.int_member);
      useIntPtr(target.ptr_member);
      useIntConstPtr(&target.int_member);
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, ValueVar) {
  RunTest(R"(
    void f(S target, const S& other) {
      useConstRef(/*const*/target);
      useVal(/*const*/target);
      useConstPtr(&target);
      useConstPtrConstRef(&target);
      /*const*/target.constMethod();
      target.nonConstMethod();
      /*const*/target(ConstTag{});
      target[42];
      /*const*/target(ConstTag{});
      target(NonConstTag{});
      useRef(target);
      usePtr(&target);
      useConstRef((/*const*/target));
      (/*const*/target).constMethod();
      (void)(/*const*/target == /*const*/target);
      (void)(/*const*/target == other);
      (void)target;
      (void)&target;
      (void)*&target;
      S copy1 = /*const*/target;
      S copy2(/*const*/target);
      useInt(/*const*/target.int_member);
      useIntConstRef(/*const*/target.int_member);
      useIntPtr(target.ptr_member);
      useIntConstPtr(&target.int_member);
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, RefVar) {
  RunTest(R"(
    void f(S& target) {
      useVal(/*const*/target);
      useConstRef(/*const*/target);
      useConstPtr(&target);
      useConstPtrConstRef(&target);
      /*const*/target.constMethod();
      target.nonConstMethod();
      /*const*/target(ConstTag{});
      target[42];
      useConstRef((/*const*/target));
      (/*const*/target).constMethod();
      (void)(/*const*/target == /*const*/target);
      (void)target;
      (void)&target;
      (void)*&target;
      S copy1 = /*const*/target;
      S copy2(/*const*/target);
      useInt(/*const*/target.int_member);
      useIntConstRef(/*const*/target.int_member);
      useIntPtr(target.ptr_member);
      useIntConstPtr(&target.int_member);
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, PtrVar) {
  RunTest(R"(
    void f(S* target) {
      useVal(*target);
      useConstRef(*target);
      useConstPtr(target);
      useConstPtrConstRef(/*const*/target);
      /*const*/target->constMethod();
      target->nonConstMethod();
      (*target)(ConstTag{});
      (*target)[42];
      target->operator[](42);
      useConstRef((*target));
      (/*const*/target)->constMethod();
      (void)(*target == *target);
      (void)*target;
      (void)target;
      S copy1 = *target;
      S copy2(*target);
      useInt(target->int_member);
      useIntConstRef(target->int_member);
      useIntPtr(target->ptr_member);
      useIntConstPtr(&target->int_member);
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, ConstPtrVar) {
  RunTest(R"(
    void f(const S* target) {
      useVal(*target);
      useConstRef(*target);
      useConstPtr(target);
      useConstPtrRef(target);
      useConstPtrPtr(&target);
      useConstPtrConstPtr(&target);
      useConstPtrConstRef(/*const*/target);
      /*const*/target->constMethod();
      (*target)(ConstTag{});
      (*target)[42];
      /*const*/target->operator[](42);
      (void)(*target == *target);
      (void)target;
      (void)*target;
      if(target) {}
      S copy1 = *target;
      S copy2(*target);
      useInt(target->int_member);
      useIntConstRef(target->int_member);
      useIntPtr(target->ptr_member);
      useIntConstPtr(&target->int_member);
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, ConstPtrPtrVar) {
  RunTest(R"(
    void f(const S** target) {
      useVal(**target);
      useConstRef(**target);
      useConstPtr(*target);
      useConstPtrRef(*target);
      useConstPtrPtr(target);
      useConstPtrConstPtr(target);
      useConstPtrConstRef(*target);
      (void)target;
      (void)*target;
      (void)**target;
      if(target) {}
      if(*target) {}
      S copy1 = **target;
      S copy2(**target);
      useInt((*target)->int_member);
      useIntConstRef((*target)->int_member);
      useIntPtr((*target)->ptr_member);
      useIntConstPtr(&(*target)->int_member);
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, ConstPtrConstPtrVar) {
  RunTest(R"(
    void f(const S* const* target) {
      useVal(**target);
      useConstRef(**target);
      useConstPtr(*target);
      useConstPtrConstPtr(target);
      useConstPtrConstRef(*target);
      (void)target;
      (void)target;
      (void)**target;
      if(target) {}
      if(*target) {}
      S copy1 = **target;
      S copy2(**target);
      useInt((*target)->int_member);
      useIntConstRef((*target)->int_member);
      useIntPtr((*target)->ptr_member);
      useIntConstPtr(&(*target)->int_member);
    }
)");
}

} // namespace test
} // namespace tidy
} // namespace clang
