// RUN: %check_clang_tidy %s google-cpp-init-class-members %t

class PositiveDefaultedDefaultConstructor {
public:
  PositiveDefaultedDefaultConstructor() = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor should initialize these fields: X

private:
  int X;
};

class PositiveDefaultedDefaultConstructorWithInitializedField {
public:
  PositiveDefaultedDefaultConstructorWithInitializedField() = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor should initialize these fields: X

private:
  int X;
  int Y = 4; // no-warning
};

class Helper {
 public:
  Helper(int x) : X(x) {}

 private:
  int X;
};

class PositiveDefaultedConstructorObjectAndPrimitive {
 public:
  PositiveDefaultedConstructorObjectAndPrimitive() = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor should initialize these fields: Y

  Helper* GetHelper() { return &X; }

  void SetY(bool enabled) { Y = enabled; }

  bool IsY() { return Y; }

 private:
  Helper X;
  bool Y;
};

struct PositiveStruct {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: these fields should be initialized: X, Y
  int X;
  int Y;
};

struct PositiveStructWithInitializedField {
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: these fields should be initialized: Y
  int X = 3; // no-warning
  int Y;
};
