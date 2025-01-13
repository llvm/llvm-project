// RUN: %check_clang_tidy %s performance-explicit-move-constructor %t

struct Empty {};

class NotReported1 {};

class NotReported2 {
public:
  NotReported2(NotReported2&&) = default;
  NotReported2(const NotReported2&) = default;
};

class NotReported3 {
public:
  explicit NotReported3(NotReported3&&) = default;
};

class NotReported4 {
public:
  explicit NotReported4(NotReported4&&) = default;
  NotReported4(const NotReported4&) = delete;
};

class NotReported5 {
public:
  explicit NotReported5(NotReported5&&) = delete;
  NotReported5(const NotReported5&) = default;
};

class Reported1 {
public:
  explicit Reported1(Reported1&&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: copy constructor may be called instead of move constructor [performance-explicit-move-constructor]
  // CHECK-FIXES: {{^  }}Reported1(Reported1&&) = default;{{$}}
  Reported1(const Reported1&) = default;
};

template <typename>
class Reported2 {
public:
  explicit Reported2(Reported2&&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: copy constructor may be called instead of move constructor [performance-explicit-move-constructor]
  // CHECK-FIXES: {{^  }}Reported2(Reported2&&) = default;{{$}}
  Reported2(const Reported2&) = default;
};

template <typename T>
class Reported3 : public T {
public:
  explicit Reported3(Reported3&&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: copy constructor may be called instead of move constructor [performance-explicit-move-constructor]
  // CHECK-FIXES: {{^  }}Reported3(Reported3&&) = default;{{$}}
  Reported3(const Reported3&) = default;
};

template <typename T>
class Reported4 {
public:
  explicit Reported4(Reported4&&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: copy constructor may be called instead of move constructor [performance-explicit-move-constructor]
  // CHECK-FIXES: {{^  }}Reported4(Reported4&&) = default;{{$}}
  Reported4(const Reported4&) = default;

  T member;
};
