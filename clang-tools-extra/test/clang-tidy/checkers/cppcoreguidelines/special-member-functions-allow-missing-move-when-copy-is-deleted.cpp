// RUN: %check_clang_tidy %s cppcoreguidelines-special-member-functions %t -- -config="{CheckOptions: {cppcoreguidelines-special-member-functions.AllowMissingMoveFunctionsWhenCopyIsDeleted: true}}" --

class DefinesEverything {
  DefinesEverything(const DefinesEverything &);
  DefinesEverything(DefinesEverything &&);
  DefinesEverything &operator=(const DefinesEverything &);
  DefinesEverything &operator=(DefinesEverything &&);
  ~DefinesEverything();
};

class DefinesNothing {
};

class DeletedCopyCtorAndOperator {
  ~DeletedCopyCtorAndOperator() = default;
  DeletedCopyCtorAndOperator(const DeletedCopyCtorAndOperator &) = delete;
  DeletedCopyCtorAndOperator &operator=(const DeletedCopyCtorAndOperator &) = delete;
};

// CHECK-MESSAGES: [[@LINE+1]]:7: warning: class 'DefaultedCopyCtorAndOperator' defines a default destructor, a copy constructor and a copy assignment operator but does not define a move constructor or a move assignment operator [cppcoreguidelines-special-member-functions]
class DefaultedCopyCtorAndOperator {
  ~DefaultedCopyCtorAndOperator() = default;
  DefaultedCopyCtorAndOperator(const DefaultedCopyCtorAndOperator &) = default;
  DefaultedCopyCtorAndOperator &operator=(const DefaultedCopyCtorAndOperator &) = default;
};

// CHECK-MESSAGES: [[@LINE+1]]:7: warning: class 'DefinedCopyCtorAndOperator' defines a default destructor, a copy constructor and a copy assignment operator but does not define a move constructor or a move assignment operator [cppcoreguidelines-special-member-functions]
class DefinedCopyCtorAndOperator {
  ~DefinedCopyCtorAndOperator() = default;
  DefinedCopyCtorAndOperator(const DefinedCopyCtorAndOperator &);
  DefinedCopyCtorAndOperator &operator=(const DefinedCopyCtorAndOperator &);
};

// CHECK-MESSAGES: [[@LINE+1]]:7: warning: class 'MissingCopyCtor' defines a default destructor and a copy assignment operator but does not define a copy constructor, a move constructor or a move assignment operator [cppcoreguidelines-special-member-functions]
class MissingCopyCtor {
  ~MissingCopyCtor() = default;
  MissingCopyCtor &operator=(const MissingCopyCtor &) = delete;
};

// CHECK-MESSAGES: [[@LINE+1]]:7:  warning: class 'MissingCopyOperator' defines a default destructor and a copy constructor but does not define a copy assignment operator, a move constructor or a move assignment operator [cppcoreguidelines-special-member-functions]
class MissingCopyOperator {
  ~MissingCopyOperator() = default;
  MissingCopyOperator(const MissingCopyOperator &) = delete;
};

// CHECK-MESSAGES: [[@LINE+1]]:7:  warning: class 'MissingAll' defines a default destructor but does not define a copy constructor, a copy assignment operator, a move constructor or a move assignment operator [cppcoreguidelines-special-member-functions]
class MissingAll {
  ~MissingAll() = default;
};
