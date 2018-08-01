
template<typename T>
class BaseTemplate {
public:
  T baseTemplateFunction();

  T baseTemplateField;

  struct NestedBaseType { };
};

template<typename T, typename S>
class TemplateClass: public BaseTemplate<T> {
public:
  T function() { return T(); }

  static void staticFunction() { }

  T field;

  struct NestedType {
    T nestedField;

    class SubNestedType {
    public:
      SubNestedType(int);
    };
    using TypeAlias = T;

    typedef int Typedef;

    enum Enum {
      EnumCase
    };
  };
};

void canonicalizeInstaniationReferences(TemplateClass<int, float> &object) {
  (void)object.function();
// CHECK1: 'c:@ST>2#T#T@TemplateClass@F@function#'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):16 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK1 %s
  (void)object.field;
// CHECK2: 'c:@ST>2#T#T@TemplateClass@FI@field'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):16 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK2 %s
  (void)object.baseTemplateFunction();
// CHECK3: 'c:@ST>1#T@BaseTemplate@F@baseTemplateFunction#'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):16 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK3 %s
  (void)object.baseTemplateField;
// CHECK4: 'c:@ST>1#T@BaseTemplate@FI@baseTemplateField'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):16 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK4 %s

  TemplateClass<int, float>::staticFunction();
// CHECK5: 'c:@ST>2#T#T@TemplateClass@F@staticFunction#S'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):30 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK5 %s

  TemplateClass<int, float>::NestedBaseType nestedBaseType;
// CHECK6: 'c:@ST>1#T@BaseTemplate@S@NestedBaseType'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):30 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK6 %s
  TemplateClass<int, float>::NestedType nestedSubType;
// CHECK7: 'c:@ST>2#T#T@TemplateClass@S@NestedType'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):30 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK7 %s
  (void)nestedSubType.nestedField;
// CHECK8: 'c:@ST>2#T#T@TemplateClass@S@NestedType@FI@nestedField'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):23 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK8 %s

  typedef TemplateClass<int, float> TT;
  TT::NestedType::SubNestedType subNestedType(0);
// CHECK9: 'c:@ST>2#T#T@TemplateClass@S@NestedType'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):7 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK9 %s
// CHECK10: 'c:@ST>2#T#T@TemplateClass@S@NestedType@S@SubNestedType'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-4):19 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK10 %s

  TT::NestedType::TypeAlias nestedTypeAlias;
// CHECK11: 'c:@ST>2#T#T@TemplateClass@S@NestedType@TypeAlias'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):19 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK11 %s
  TT::NestedType::Typedef nestedTypedef;
// CHECK12: 'c:{{.*}}CanonicalizeInstantiatedDecls.cpp@ST>2#T#T@TemplateClass@S@NestedType@T@Typedef'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):19 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK12 %s

  TT::NestedType::Enum nestedEnum;
// CHECK13: 'c:@ST>2#T#T@TemplateClass@S@NestedType@E@Enum'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):19 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK13 %s
  (void)TT::NestedType::Enum::EnumCase;
// CHECK14: 'c:@ST>2#T#T@TemplateClass@S@NestedType@E@Enum@EnumCase'
// RUN: clang-refactor-test rename-initiate -at=%s:%(line-2):31 -new-name=x -dump-symbols %s | FileCheck --check-prefix=CHECK14 %s
}
