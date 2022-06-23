// RUN: %check_clang_tidy %s cppcoreguidelines-special-member-functions %t

class DefinesDestructor {
  ~DefinesDestructor();
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'DefinesDestructor' defines a non-default destructor but does not define a copy constructor, a copy assignment operator, a move constructor or a move assignment operator [cppcoreguidelines-special-member-functions]

class DefinesDefaultedDestructor {
  ~DefinesDefaultedDestructor() = default;
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'DefinesDefaultedDestructor' defines a default destructor but does not define a copy constructor, a copy assignment operator, a move constructor or a move assignment operator [cppcoreguidelines-special-member-functions]

class DefinesCopyConstructor {
  DefinesCopyConstructor(const DefinesCopyConstructor &);
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'DefinesCopyConstructor' defines a copy constructor but does not define a destructor, a copy assignment operator, a move constructor or a move assignment operator [cppcoreguidelines-special-member-functions]

class DefinesCopyAssignment {
  DefinesCopyAssignment &operator=(const DefinesCopyAssignment &);
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'DefinesCopyAssignment' defines a copy assignment operator but does not define a destructor, a copy constructor, a move constructor or a move assignment operator [cppcoreguidelines-special-member-functions]

class DefinesMoveConstructor {
  DefinesMoveConstructor(DefinesMoveConstructor &&);
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'DefinesMoveConstructor' defines a move constructor but does not define a destructor, a copy constructor, a copy assignment operator or a move assignment operator [cppcoreguidelines-special-member-functions]

class DefinesMoveAssignment {
  DefinesMoveAssignment &operator=(DefinesMoveAssignment &&);
};
// CHECK-MESSAGES: [[@LINE-3]]:7: warning: class 'DefinesMoveAssignment' defines a move assignment operator but does not define a destructor, a copy constructor, a copy assignment operator or a move constructor [cppcoreguidelines-special-member-functions]
class DefinesNothing {
};

class DefinesEverything {
  DefinesEverything(const DefinesEverything &);
  DefinesEverything &operator=(const DefinesEverything &);
  DefinesEverything(DefinesEverything &&);
  DefinesEverything &operator=(DefinesEverything &&);
  ~DefinesEverything();
};

class DeletesEverything {
  DeletesEverything(const DeletesEverything &) = delete;
  DeletesEverything &operator=(const DeletesEverything &) = delete;
  DeletesEverything(DeletesEverything &&) = delete;
  DeletesEverything &operator=(DeletesEverything &&) = delete;
  ~DeletesEverything() = delete;
};

class DeletesCopyDefaultsMove {
  DeletesCopyDefaultsMove(const DeletesCopyDefaultsMove &) = delete;
  DeletesCopyDefaultsMove &operator=(const DeletesCopyDefaultsMove &) = delete;
  DeletesCopyDefaultsMove(DeletesCopyDefaultsMove &&) = default;
  DeletesCopyDefaultsMove &operator=(DeletesCopyDefaultsMove &&) = default;
  ~DeletesCopyDefaultsMove() = default;
};

template <typename T>
struct TemplateClass {
  TemplateClass() = default;
  TemplateClass(const TemplateClass &);
  TemplateClass &operator=(const TemplateClass &);
  TemplateClass(TemplateClass &&);
  TemplateClass &operator=(TemplateClass &&);
  ~TemplateClass();
};

// Multiple instantiations of a class template will trigger multiple matches for defined special members.
// This should not cause problems.
TemplateClass<int> InstantiationWithInt;
TemplateClass<double> InstantiationWithDouble;
