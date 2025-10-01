// RUN: %check_clang_tidy %s cppcoreguidelines-special-member-functions %t -- -config="{CheckOptions: {cppcoreguidelines-special-member-functions.IgnoreMacros: false}}" --

class DefinesDestructor {
  ~DefinesDestructor();
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'DefinesDestructor' defines a destructor but does not define a copy constructor, a copy assignment operator, a move constructor or a move assignment operator [cppcoreguidelines-special-member-functions]

class DefinesDefaultedDestructor {
  ~DefinesDefaultedDestructor() = default;
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'DefinesDefaultedDestructor' defines a default destructor but does not define a copy constructor, a copy assignment operator, a move constructor or a move assignment operator [cppcoreguidelines-special-member-functions]

class DefinesCopyConstructor {
  DefinesCopyConstructor(const DefinesCopyConstructor &);
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'DefinesCopyConstructor' defines a copy constructor but does not define a destructor, a copy assignment operator, a move constructor or a move assignment operator [cppcoreguidelines-special-member-functions]

class DefinesNothing {
};

class DefinesEverything {
  DefinesEverything(const DefinesEverything &);
  DefinesEverything &operator=(const DefinesEverything &);
  DefinesEverything(DefinesEverything &&);
  DefinesEverything &operator=(DefinesEverything &&);
  ~DefinesEverything();
};

#define DEFINE_DESTRUCTOR_ONLY(ClassName) \
  class ClassName { \
    ~ClassName(); \
  };

#define DEFINE_COPY_CTOR_ONLY(ClassName) \
  class ClassName { \
    ClassName(const ClassName &); \
  };

#define DEFINE_CLASS_WITH_DTOR(ClassName) \
  class ClassName { \
    ~ClassName(); \
  };

DEFINE_DESTRUCTOR_ONLY(MacroDefinedClass1)
// CHECK-MESSAGES: [[@LINE-1]]:24: warning: class 'MacroDefinedClass1' defines a destructor but does not define a copy constructor, a copy assignment operator, a move constructor or a move assignment operator
DEFINE_COPY_CTOR_ONLY(MacroDefinedClass2)
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: class 'MacroDefinedClass2' defines a copy constructor but does not define a destructor, a copy assignment operator, a move constructor or a move assignment operator
DEFINE_CLASS_WITH_DTOR(MacroDefinedClass3)
// CHECK-MESSAGES: [[@LINE-1]]:24: warning: class 'MacroDefinedClass3' defines a destructor but does not define a copy constructor, a copy assignment operator, a move constructor or a move assignment operator

// Test partial macro expansion
#define CLASS_NAME MacroNamedClass
class CLASS_NAME {
  ~MacroNamedClass();
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'MacroNamedClass' defines a destructor but does not define a copy constructor, a copy assignment operator, a move constructor or a move assignment operator

