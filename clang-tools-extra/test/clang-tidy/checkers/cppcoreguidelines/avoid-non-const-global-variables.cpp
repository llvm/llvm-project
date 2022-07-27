// RUN: %check_clang_tidy %s cppcoreguidelines-avoid-non-const-global-variables %t

int nonConstInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'nonConstInt' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

int &nonConstIntReference = nonConstInt;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: variable 'nonConstIntReference' provides global access to a non-const object; consider making the referenced data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

int *pointerToNonConstInt = &nonConstInt;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: variable 'pointerToNonConstInt' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]
// CHECK-MESSAGES: :[[@LINE-2]]:6: warning: variable 'pointerToNonConstInt' provides global access to a non-const object; consider making the pointed-to data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

int *const constPointerToNonConstInt = &nonConstInt;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'constPointerToNonConstInt' provides global access to a non-const object; consider making the pointed-to data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

namespace namespace_name {
int nonConstNamespaceInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'nonConstNamespaceInt' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

const int constNamespaceInt = 0;
} // namespace namespace_name

const int constInt = 0;

const int *pointerToConstInt = &constInt;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'pointerToConstInt' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

const int *const constPointerToConstInt = &constInt;

const int &constReferenceToConstInt = constInt;

constexpr int constexprInt = 0;

int function() {
  int nonConstReturnValue = 0;
  return nonConstReturnValue;
}

namespace {
int nonConstAnonymousNamespaceInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'nonConstAnonymousNamespaceInt' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]
} // namespace

class DummyClass {
public:
  int nonConstPublicMemberVariable = 0;
  const int constPublicMemberVariable = 0;

private:
  int nonConstPrivateMemberVariable = 0;
  const int constPrivateMemberVariable = 0;
};

DummyClass nonConstClassInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'nonConstClassInstance' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

DummyClass *pointerToNonConstDummyClass = &nonConstClassInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'pointerToNonConstDummyClass' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]
// CHECK-MESSAGES: :[[@LINE-2]]:13: warning: variable 'pointerToNonConstDummyClass' provides global access to a non-const object; consider making the pointed-to data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

DummyClass &referenceToNonConstDummyClass = nonConstClassInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'referenceToNonConstDummyClass' provides global access to a non-const object; consider making the referenced data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

int *nonConstPointerToMember = &nonConstClassInstance.nonConstPublicMemberVariable;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: variable 'nonConstPointerToMember' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]
// CHECK-MESSAGES: :[[@LINE-2]]:6: warning: variable 'nonConstPointerToMember' provides global access to a non-const object; consider making the pointed-to data 'const' [cppcoreguidelines-avoid-non-const-global-variables]
int *const constPointerToNonConstMember = &nonConstClassInstance.nonConstPublicMemberVariable;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'constPointerToNonConstMember' provides global access to a non-const object; consider making the pointed-to data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

const DummyClass constClassInstance;

DummyClass *const constPointerToNonConstDummyClass = &nonConstClassInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: variable 'constPointerToNonConstDummyClass' provides global access to a non-const object; consider making the pointed-to data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

const DummyClass *nonConstPointerToConstDummyClass = &constClassInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: variable 'nonConstPointerToConstDummyClass' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

const DummyClass *const constPointerToConstDummyClass = &constClassInstance;

const int *const constPointerToConstMember = &constClassInstance.nonConstPublicMemberVariable;

const DummyClass &constReferenceToDummyClass = constClassInstance;

namespace namespace_name {
DummyClass nonConstNamespaceClassInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'nonConstNamespaceClassInstance' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

const DummyClass constDummyClassInstance;
} // namespace namespace_name

// CHECKING FOR NON-CONST GLOBAL ENUM /////////////////////////////////////////
enum DummyEnum {
  first,
  second
};

DummyEnum nonConstDummyEnumInstance = DummyEnum::first;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: variable 'nonConstDummyEnumInstance' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

DummyEnum *pointerToNonConstDummyEnum = &nonConstDummyEnumInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'pointerToNonConstDummyEnum' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]
// CHECK-MESSAGES: :[[@LINE-2]]:12: warning: variable 'pointerToNonConstDummyEnum' provides global access to a non-const object; consider making the pointed-to data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

DummyEnum &referenceToNonConstDummyEnum = nonConstDummyEnumInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'referenceToNonConstDummyEnum' provides global access to a non-const object; consider making the referenced data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

DummyEnum *const constPointerToNonConstDummyEnum = &nonConstDummyEnumInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: variable 'constPointerToNonConstDummyEnum' provides global access to a non-const object; consider making the pointed-to data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

const DummyEnum constDummyEnumInstance = DummyEnum::first;

const DummyEnum *nonConstPointerToConstDummyEnum = &constDummyEnumInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: variable 'nonConstPointerToConstDummyEnum' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

const DummyEnum *const constPointerToConstDummyEnum = &constDummyEnumInstance;

const DummyEnum &referenceToConstDummyEnum = constDummyEnumInstance;

namespace namespace_name {
DummyEnum nonConstNamespaceEnumInstance = DummyEnum::first;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: variable 'nonConstNamespaceEnumInstance' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

const DummyEnum constNamespaceEnumInstance = DummyEnum::first;
} // namespace namespace_name

namespace {
DummyEnum nonConstAnonymousNamespaceEnumInstance = DummyEnum::first;
}
// CHECK-MESSAGES: :[[@LINE-2]]:11: warning: variable 'nonConstAnonymousNamespaceEnumInstance' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

// CHECKING FOR NON-CONST GLOBAL STRUCT ///////////////////////////////////////
struct DummyStruct {
public:
  int structIntElement = 0;
  const int constStructIntElement = 0;

private:
  int privateStructIntElement = 0;
};

DummyStruct nonConstDummyStructInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'nonConstDummyStructInstance' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

DummyStruct *pointerToNonConstDummyStruct = &nonConstDummyStructInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: variable 'pointerToNonConstDummyStruct' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]
// CHECK-MESSAGES: :[[@LINE-2]]:14: warning: variable 'pointerToNonConstDummyStruct' provides global access to a non-const object; consider making the pointed-to data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

DummyStruct &referenceToNonConstDummyStruct = nonConstDummyStructInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: variable 'referenceToNonConstDummyStruct' provides global access to a non-const object; consider making the referenced data 'const' [cppcoreguidelines-avoid-non-const-global-variables]
DummyStruct *const constPointerToNonConstDummyStruct = &nonConstDummyStructInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: variable 'constPointerToNonConstDummyStruct' provides global access to a non-const object; consider making the pointed-to data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

const DummyStruct constDummyStructInstance;

const DummyStruct *nonConstPointerToConstDummyStruct = &constDummyStructInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: variable 'nonConstPointerToConstDummyStruct' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

const DummyStruct *const constPointerToConstDummyStruct = &constDummyStructInstance;

const DummyStruct &referenceToConstDummyStruct = constDummyStructInstance;

namespace namespace_name {
DummyStruct nonConstNamespaceDummyStructInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'nonConstNamespaceDummyStructInstance' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

const DummyStruct constNamespaceDummyStructInstance;
} // namespace namespace_name

namespace {
DummyStruct nonConstAnonymousNamespaceStructInstance;
}
// CHECK-MESSAGES: :[[@LINE-2]]:13: warning: variable 'nonConstAnonymousNamespaceStructInstance' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

// CHECKING FOR NON-CONST GLOBAL UNION ////////////////////////////////////////
union DummyUnion {
  int unionInteger;
  char unionChar;
};

DummyUnion nonConstUnionIntInstance = {0x0};
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'nonConstUnionIntInstance' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

DummyUnion *nonConstPointerToNonConstUnionInt = &nonConstUnionIntInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'nonConstPointerToNonConstUnionInt' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]
// CHECK-MESSAGES: :[[@LINE-2]]:13: warning: variable 'nonConstPointerToNonConstUnionInt' provides global access to a non-const object; consider making the pointed-to data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

DummyUnion *const constPointerToNonConstUnionInt = &nonConstUnionIntInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: variable 'constPointerToNonConstUnionInt' provides global access to a non-const object; consider making the pointed-to data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

DummyUnion &referenceToNonConstUnionInt = nonConstUnionIntInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'referenceToNonConstUnionInt' provides global access to a non-const object; consider making the referenced data 'const' [cppcoreguidelines-avoid-non-const-global-variables]

const DummyUnion constUnionIntInstance = {0x0};

const DummyUnion *nonConstPointerToConstUnionInt = &constUnionIntInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: variable 'nonConstPointerToConstUnionInt' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

const DummyUnion *const constPointerToConstUnionInt = &constUnionIntInstance;

const DummyUnion &referenceToConstUnionInt = constUnionIntInstance;

namespace namespace_name {
DummyUnion nonConstNamespaceDummyUnionInstance;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'nonConstNamespaceDummyUnionInstance' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

const DummyUnion constNamespaceDummyUnionInstance = {0x0};
} // namespace namespace_name

namespace {
DummyUnion nonConstAnonymousNamespaceUnionInstance = {0x0};
}
// CHECK-MESSAGES: :[[@LINE-2]]:12: warning: variable 'nonConstAnonymousNamespaceUnionInstance' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

// CHECKING FOR NON-CONST GLOBAL FUNCTION POINTER /////////////////////////////
int dummyFunction() {
  return 0;
}

typedef int (*functionPointer)();
functionPointer fp1 = &dummyFunction;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: variable 'fp1' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

typedef int (*const functionConstPointer)();
functionPointer fp2 = &dummyFunction;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: variable 'fp2' is non-const and globally accessible, consider making it const [cppcoreguidelines-avoid-non-const-global-variables]

// CHECKING FOR NON-CONST GLOBAL TEMPLATE VARIABLE ////////////////////////////
template <class T>
constexpr T templateVariable = T(0L);

// CHECKING AGAINST FALSE POSITIVES INSIDE FUNCTION SCOPE /////////////////////
int main() {
  for (int i = 0; i < 3; ++i) {
    static int staticNonConstLoopVariable = 42;
    int nonConstLoopVariable = 42;
    nonConstInt = nonConstLoopVariable + i + staticNonConstLoopVariable;
  }
}
