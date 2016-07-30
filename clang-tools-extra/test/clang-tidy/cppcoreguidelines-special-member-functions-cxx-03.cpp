// RUN: %check_clang_tidy %s cppcoreguidelines-special-member-functions %t -- -- -std=c++03

class DefinesDestructor {
  ~DefinesDestructor();
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'DefinesDestructor' defines a destructor but does not define a copy constructor or a copy assignment operator [cppcoreguidelines-special-member-functions]

class DefinesCopyConstructor {
  DefinesCopyConstructor(const DefinesCopyConstructor &);
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'DefinesCopyConstructor' defines a copy constructor but does not define a destructor or a copy assignment operator [cppcoreguidelines-special-member-functions]

class DefinesCopyAssignment {
  DefinesCopyAssignment &operator=(const DefinesCopyAssignment &);
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'DefinesCopyAssignment' defines a copy assignment operator but does not define a destructor or a copy constructor [cppcoreguidelines-special-member-functions]

class DefinesNothing {
};

class DefinesEverything {
  DefinesEverything(const DefinesEverything &);
  DefinesEverything &operator=(const DefinesEverything &);
  ~DefinesEverything();
};

