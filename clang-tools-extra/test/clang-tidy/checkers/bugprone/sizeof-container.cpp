// RUN: %check_clang_tidy %s bugprone-sizeof-container %t -- -- -target x86_64-unknown-unknown

namespace std {

typedef unsigned int size_t;

template <typename T>
struct basic_string {
  size_t size() const;
};

template <typename T>
basic_string<T> operator+(const basic_string<T> &, const T *);

typedef basic_string<char> string;

template <typename T>
struct vector {
  size_t size() const;
};

// std::bitset<> is not a container. sizeof() is reasonable for it.
template <size_t N>
struct bitset {
  size_t size() const;
};

// std::array<> is, well, an array. sizeof() is reasonable for it.
template <typename T, size_t N>
struct array {
  size_t size() const;
};

class fake_container1 {
  size_t size() const; // non-public
};

struct fake_container2 {
  size_t size(); // non-const
};

}

using std::size_t;

#define ARRAYSIZE(a) \
  ((sizeof(a) / sizeof(*(a))) / static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))

#define ARRAYSIZE2(a) \
  (((sizeof(a)) / (sizeof(*(a)))) / static_cast<size_t>(!((sizeof(a)) % (sizeof(*(a))))))

struct string {
  std::size_t size() const;
};

template<typename T>
void g(T t) {
  (void)sizeof(t);
}

void f() {
  string s1;
  std::string s2;
  std::vector<int> v;

  int a = 42 + sizeof(s1);
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: sizeof() doesn't return the size of the container; did you mean .size()? [bugprone-sizeof-container]
  a = 123 * sizeof(s2);
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: sizeof() doesn't return the size
  a = 45 + sizeof(s2 + "asdf");
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: sizeof() doesn't return the size
  a = sizeof(v);
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: sizeof() doesn't return the size
  a = sizeof(std::vector<int>{});
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: sizeof() doesn't return the size

  a = sizeof(a);
  a = sizeof(int);
  a = sizeof(std::string);
  a = sizeof(std::vector<int>);

  g(s1);
  g(s2);
  g(v);

  std::fake_container1 fake1;
  std::fake_container2 fake2;
  std::bitset<7> std_bitset;
  std::array<int, 3> std_array;

  a = sizeof(fake1);
  a = sizeof(fake2);
  a = sizeof(std_bitset);
  a = sizeof(std_array);


  std::string arr[3];
  a = ARRAYSIZE(arr);
  a = ARRAYSIZE2(arr);
  a = sizeof(arr) / sizeof(arr[0]);

  (void)a;
}
