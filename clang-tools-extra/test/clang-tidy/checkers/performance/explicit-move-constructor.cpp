// RUN: %check_clang_tidy %s performance-explicit-move-constructor %t

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

class Reported {
public:
  explicit Reported(Reported&&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: copy constructor may be called instead of move constructor [performance-explicit-move-constructor]
  // CHECK-FIXES: {{^  }}Reported(Reported&&) = default;{{$}}
  Reported(const Reported&) = default;
};
