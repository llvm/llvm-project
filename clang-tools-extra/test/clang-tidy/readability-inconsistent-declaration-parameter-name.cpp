// RUN: %check_clang_tidy %s readability-inconsistent-declaration-parameter-name %t -- -- -std=c++11 -fno-delayed-template-parsing

void consistentFunction(int a, int b, int c);
void consistentFunction(int a, int b, int c);
void consistentFunction(int a, int b, int /*c*/);
void consistentFunction(int /*c*/, int /*c*/, int /*c*/);

//////////////////////////////////////////////////////

// CHECK-MESSAGES: :[[@LINE+1]]:6: warning: function 'inconsistentFunction' has 2 other declarations with different parameter names [readability-inconsistent-declaration-parameter-name]
void inconsistentFunction(int a, int b, int c);
// CHECK-MESSAGES: :[[@LINE+2]]:6: note: the 1st inconsistent declaration seen here
// CHECK-MESSAGES: :[[@LINE+1]]:6: note: differing parameters are named here: ('d', 'e', 'f'), in the other declaration: ('a', 'b', 'c')
void inconsistentFunction(int d, int e, int f);
// CHECK-MESSAGES: :[[@LINE+2]]:6: note: the 2nd inconsistent declaration seen here
// CHECK-MESSAGES: :[[@LINE+1]]:6: note: differing parameters are named here: ('x', 'y', 'z'), in the other declaration: ('a', 'b', 'c')
void inconsistentFunction(int x, int y, int z);

//////////////////////////////////////////////////////

// CHECK-MESSAGES: :[[@LINE+4]]:6: warning: function 'inconsistentFunctionWithVisibleDefinition' has a definition with different parameter names [readability-inconsistent-declaration-parameter-name]
// CHECK-MESSAGES: :[[@LINE+9]]:6: note: the definition seen here
// CHECK-MESSAGES: :[[@LINE+2]]:6: note: differing parameters are named here: ('a'), in definition: ('c')
// CHECK-FIXES: void inconsistentFunctionWithVisibleDefinition(int c);
void inconsistentFunctionWithVisibleDefinition(int a);
// CHECK-MESSAGES: :[[@LINE+4]]:6: warning: function 'inconsistentFunctionWithVisibleDefinition' has a definition
// CHECK-MESSAGES: :[[@LINE+4]]:6: note: the definition seen here
// CHECK-MESSAGES: :[[@LINE+2]]:6: note: differing parameters are named here: ('b'), in definition: ('c')
// CHECK-FIXES: void inconsistentFunctionWithVisibleDefinition(int c);
void inconsistentFunctionWithVisibleDefinition(int b);
void inconsistentFunctionWithVisibleDefinition(int c) { c; }

// CHECK-MESSAGES: :[[@LINE+3]]:6: warning: function 'inconsidentFunctionWithUnreferencedParameterInDefinition' has a definition
// CHECK-MESSAGES: :[[@LINE+3]]:6: note: the definition seen here
// CHECK-MESSAGES: :[[@LINE+1]]:6: note: differing parameters are named here: ('a'), in definition: ('b')
void inconsidentFunctionWithUnreferencedParameterInDefinition(int a);
void inconsidentFunctionWithUnreferencedParameterInDefinition(int b) {}

//////////////////////////////////////////////////////

struct Struct {
// CHECK-MESSAGES: :[[@LINE+4]]:8: warning: function 'Struct::inconsistentFunction' has a definition
// CHECK-MESSAGES: :[[@LINE+6]]:14: note: the definition seen here
// CHECK-MESSAGES: :[[@LINE+2]]:8: note: differing parameters are named here: ('a'), in definition: ('b')
// CHECK-FIXES: void inconsistentFunction(int b);
  void inconsistentFunction(int a);
};

void Struct::inconsistentFunction(int b) { b = 0; }

//////////////////////////////////////////////////////

struct SpecialFunctions {
// CHECK-MESSAGES: :[[@LINE+4]]:3: warning: function 'SpecialFunctions::SpecialFunctions' has a definition
// CHECK-MESSAGES: :[[@LINE+12]]:19: note: the definition seen here
// CHECK-MESSAGES: :[[@LINE+2]]:3: note: differing parameters are named here: ('a'), in definition: ('b')
// CHECK-FIXES: SpecialFunctions(int b);
  SpecialFunctions(int a);

// CHECK-MESSAGES: :[[@LINE+4]]:21: warning: function 'SpecialFunctions::operator=' has a definition
// CHECK-MESSAGES: :[[@LINE+8]]:37: note: the definition seen here
// CHECK-MESSAGES: :[[@LINE+2]]:21: note: differing parameters are named here: ('a'), in definition: ('b')
// CHECK-FIXES: SpecialFunctions& operator=(const SpecialFunctions& b);
  SpecialFunctions& operator=(const SpecialFunctions& a);
};

SpecialFunctions::SpecialFunctions(int b) { b; }

SpecialFunctions& SpecialFunctions::operator=(const SpecialFunctions& b) { b; return *this; }

//////////////////////////////////////////////////////

// CHECK-MESSAGES: :[[@LINE+5]]:6: warning: function 'templateFunctionWithSeparateDeclarationAndDefinition' has a definition
// CHECK-MESSAGES: :[[@LINE+7]]:6: note: the definition seen here
// CHECK-MESSAGES: :[[@LINE+3]]:6: note: differing parameters are named here: ('a'), in definition: ('b')
// CHECK-FIXES: void templateFunctionWithSeparateDeclarationAndDefinition(T b);
template<typename T>
void templateFunctionWithSeparateDeclarationAndDefinition(T a);

template<typename T>
void templateFunctionWithSeparateDeclarationAndDefinition(T b) { b; }

//////////////////////////////////////////////////////

template<typename T>
void templateFunctionWithSpecializations(T a) { a; }

template<>
// CHECK-MESSAGES: :[[@LINE+3]]:6: warning: function template specialization 'templateFunctionWithSpecializations<int>' has a primary template declaration with different parameter names [readability-inconsistent-declaration-parameter-name]
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: the primary template declaration seen here
// CHECK-MESSAGES: :[[@LINE+1]]:6: note: differing parameters are named here: ('b'), in primary template declaration: ('a')
void templateFunctionWithSpecializations(int b) { b; }

template<>
// CHECK-MESSAGES: :[[@LINE+3]]:6: warning: function template specialization 'templateFunctionWithSpecializations<float>' has a primary template
// CHECK-MESSAGES: :[[@LINE-10]]:6: note: the primary template declaration seen here
// CHECK-MESSAGES: :[[@LINE+1]]:6: note: differing parameters are named here: ('c'), in primary template declaration: ('a')
void templateFunctionWithSpecializations(float c) { c; }

//////////////////////////////////////////////////////

template<typename T>
void templateFunctionWithoutDefinitionButWithSpecialization(T a);

template<>
// CHECK-MESSAGES: :[[@LINE+3]]:6: warning: function template specialization 'templateFunctionWithoutDefinitionButWithSpecialization<int>' has a primary template
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: the primary template declaration seen here
// CHECK-MESSAGES: :[[@LINE+1]]:6: note: differing parameters are named here: ('b'), in primary template declaration: ('a')
void templateFunctionWithoutDefinitionButWithSpecialization(int b) { b; }

//////////////////////////////////////////////////////

template<typename T>
void templateFunctionWithSeparateSpecializationDeclarationAndDefinition(T a);

template<>
// CHECK-MESSAGES: :[[@LINE+3]]:6: warning: function template specialization 'templateFunctionWithSeparateSpecializationDeclarationAndDefinition<int>' has a primary template
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: the primary template declaration seen here
// CHECK-MESSAGES: :[[@LINE+1]]:6: note: differing parameters are named here: ('b'), in primary template declaration: ('a')
void templateFunctionWithSeparateSpecializationDeclarationAndDefinition(int b);

template<>
// CHECK-MESSAGES: :[[@LINE+3]]:6: warning: function template specialization 'templateFunctionWithSeparateSpecializationDeclarationAndDefinition<int>' has a primary template
// CHECK-MESSAGES: :[[@LINE-10]]:6: note: the primary template declaration seen here
// CHECK-MESSAGES: :[[@LINE+1]]:6: note: differing parameters are named here: ('c'), in primary template declaration: ('a')
void templateFunctionWithSeparateSpecializationDeclarationAndDefinition(int c) { c; }

//////////////////////////////////////////////////////

template<typename T>
class ClassTemplate
{
public:
// CHECK-MESSAGES: :[[@LINE+4]]:10: warning: function 'ClassTemplate::functionInClassTemplateWithSeparateDeclarationAndDefinition' has a definition
// CHECK-MESSAGES: :[[@LINE+7]]:24: note: the definition seen here
// CHECK-MESSAGES: :[[@LINE+2]]:10: note: differing parameters are named here: ('a'), in definition: ('b')
// CHECK-FIXES: void functionInClassTemplateWithSeparateDeclarationAndDefinition(int b);
    void functionInClassTemplateWithSeparateDeclarationAndDefinition(int a);
};

template<typename T>
void ClassTemplate<T>::functionInClassTemplateWithSeparateDeclarationAndDefinition(int b) { b; }

//////////////////////////////////////////////////////

class Class
{
public:
    template<typename T>
// CHECK-MESSAGES: :[[@LINE+4]]:8: warning: function 'Class::memberFunctionTemplateWithSeparateDeclarationAndDefinition' has a definition
// CHECK-MESSAGES: :[[@LINE+12]]:13: note: the definition seen here
// CHECK-MESSAGES: :[[@LINE+2]]:8: note: differing parameters are named here: ('a'), in definition: ('b')
// CHECK-FIXES: void memberFunctionTemplateWithSeparateDeclarationAndDefinition(T b);
  void memberFunctionTemplateWithSeparateDeclarationAndDefinition(T a);

  template<typename T>
  void memberFunctionTemplateWithSpecializations(T a) { a; }
};

//////////////////////////////////////////////////////

template<typename T>
void Class::memberFunctionTemplateWithSeparateDeclarationAndDefinition(T b) { b; }

//////////////////////////////////////////////////////

template<>
// CHECK-MESSAGES: :[[@LINE+3]]:13: warning: function template specialization 'Class::memberFunctionTemplateWithSpecializations<int>' has a primary template
// CHECK-MESSAGES: :[[@LINE-12]]:8: note: the primary template declaration seen here
// CHECK-MESSAGES: :[[@LINE+1]]:13: note: differing parameters are named here: ('b'), in primary template declaration: ('a')
void Class::memberFunctionTemplateWithSpecializations(int b) { b; }

template<>
// CHECK-MESSAGES: :[[@LINE+3]]:13: warning: function template specialization 'Class::memberFunctionTemplateWithSpecializations<float>' has a primary template
// CHECK-MESSAGES: :[[@LINE-18]]:8: note: the primary template declaration seen here
// CHECK-MESSAGES: :[[@LINE+1]]:13: note: differing parameters are named here: ('c'), in primary template declaration: ('a')
void Class::memberFunctionTemplateWithSpecializations(float c) { c; }

//////////////////////////////////////////////////////

// This resulted in a warning by default.
#define MACRO() \
  void f(int x);

struct S {
  MACRO();
};

void S::f(int y)
{
}
