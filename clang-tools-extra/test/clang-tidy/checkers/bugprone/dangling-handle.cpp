// RUN: %check_clang_tidy -std=c++11,c++14 %s bugprone-dangling-handle %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             [{key: bugprone-dangling-handle.HandleClasses, \
// RUN:               value: 'std::basic_string_view; ::llvm::StringRef;'}]}"
// FIXME: Fix the checker to work in C++17 or later mode.

namespace std {

template <typename T>
class vector {
 public:
  using const_iterator = const T*;
  using iterator = T*;
  using size_type = int;

  void assign(size_type count, const T& value);
  iterator insert(const_iterator pos, const T& value);
  iterator insert(const_iterator pos, T&& value);
  iterator insert(const_iterator pos, size_type count, const T& value);
  void push_back(const T&);
  void push_back(T&&);
  void resize(size_type count, const T& value);
};

template <typename, typename>
class pair {};

template <typename T>
class set {
 public:
  using const_iterator = const T*;
  using iterator = T*;

  std::pair<iterator, bool> insert(const T& value);
  std::pair<iterator, bool> insert(T&& value);
  iterator insert(const_iterator hint, const T& value);
  iterator insert(const_iterator hint, T&& value);
};

template <typename Key, typename Value>
class map {
 public:
  using value_type = pair<Key, Value>;
  value_type& operator[](const Key& key);
  value_type& operator[](Key&& key);
};

class basic_string_view;

class basic_string {
 public:
  basic_string();
  basic_string(const char*);

  typedef basic_string_view str_view;
  operator str_view() const noexcept;

  ~basic_string();
};

typedef basic_string string;

class basic_string_view {
 public:
  basic_string_view(const char*);
};

typedef basic_string_view string_view;

}  // namespace std

namespace llvm {

class StringRef {
 public:
  StringRef();
  StringRef(const char*);
  StringRef(const std::string&);
};

}  // namespace llvm

std::string ReturnsAString();

void Positives() {
  std::string_view view1 = std::string();
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: std::basic_string_view outlives its value [bugprone-dangling-handle]

  std::string_view view_2 = ReturnsAString();
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: std::basic_string_view outlives

  view1 = std::string();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: std::basic_string_view outlives

  const std::string& str_ref = "";
  std::string_view view3 = true ? "A" : str_ref;
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: std::basic_string_view outlives
  view3 = true ? "A" : str_ref;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: std::basic_string_view outlives

  std::string_view view4(ReturnsAString());
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: std::basic_string_view outlives
}

void OtherTypes() {
  llvm::StringRef ref = std::string();
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: llvm::StringRef outlives its value
}

const char static_array[] = "A";
std::string_view ReturnStatements(int i, std::string value_arg,
                                  const std::string &ref_arg) {
  const char array[] = "A";
  const char* ptr = "A";
  std::string s;
  static std::string ss;
  switch (i) {
    // Bad cases
    case 0:
      return array;  // refers to local
      // CHECK-MESSAGES: [[@LINE-1]]:7: warning: std::basic_string_view outliv
    case 1:
      return s;  // refers to local
      // CHECK-MESSAGES: [[@LINE-1]]:7: warning: std::basic_string_view outliv
    case 2:
      return std::string();  // refers to temporary
      // CHECK-MESSAGES: [[@LINE-1]]:7: warning: std::basic_string_view outliv
    case 3:
      return value_arg;  // refers to by-value arg
      // CHECK-MESSAGES: [[@LINE-1]]:7: warning: std::basic_string_view outliv

    // Ok cases
    case 100:
      return ss;  // refers to static
    case 101:
      return static_array;  // refers to static
    case 102:
      return ptr;  // pointer is ok
    case 103:
      return ref_arg;  // refers to by-ref arg
  }

  struct S {
    std::string_view view() { return value; }
    std::string value;
  };

  (void)[&]()->std::string_view {
    // This should not warn. The string is bound by reference.
    return s;
  };
  (void)[=]() -> std::string_view {
    // This should not warn. The reference is valid as long as the lambda.
    return s;
  };
  (void)[=]() -> std::string_view {
    // FIXME: This one should warn. We are returning a reference to a local
    // lambda variable.
    std::string local;
    return local;
  };
  return "";
}

void Containers() {
  std::vector<std::string_view> v;
  v.assign(3, std::string());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: std::basic_string_view outlives
  v.insert(nullptr, std::string());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: std::basic_string_view outlives
  v.insert(nullptr, 3, std::string());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: std::basic_string_view outlives
  v.push_back(std::string());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: std::basic_string_view outlives
  v.resize(3, std::string());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: std::basic_string_view outlives

  std::set<std::string_view> s;
  s.insert(std::string());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: std::basic_string_view outlives
  s.insert(nullptr, std::string());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: std::basic_string_view outlives

  std::map<std::string_view, int> m;
  m[std::string()];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: std::basic_string_view outlives
}

void TakesAStringView(std::string_view);

void Negatives(std::string_view default_arg = ReturnsAString()) {
  std::string str;
  std::string_view view = str;

  TakesAStringView(std::string());
}

