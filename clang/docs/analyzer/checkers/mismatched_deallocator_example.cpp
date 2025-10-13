// C, C++
void test() {
  int *p = (int *)malloc(sizeof(int));
  delete p; // warn
}

// C, C++
void __attribute((ownership_returns(malloc))) *user_malloc(size_t);
void __attribute((ownership_takes(malloc, 1))) *user_free(void *);

void __attribute((ownership_returns(malloc1))) *user_malloc1(size_t);
void __attribute((ownership_takes(malloc1, 1))) *user_free1(void *);

void test() {
  int *p = (int *)user_malloc(sizeof(int));
  delete p; // warn
}

// C, C++
void test() {
  int *p = new int;
  free(p); // warn
}

// C, C++
void test() {
  int *p = new int[1];
  realloc(p, sizeof(long)); // warn
}

// C, C++
void test() {
  int *p = user_malloc(10);
  user_free1(p); // warn
}

// C, C++
template <typename T>
struct SimpleSmartPointer {
  T *ptr;

  explicit SimpleSmartPointer(T *p = 0) : ptr(p) {}
  ~SimpleSmartPointer() {
    delete ptr; // warn
  }
};

void test() {
  SimpleSmartPointer<int> a((int *)malloc(4));
}

// C++
void test() {
  int *p = (int *)operator new(0);
  delete[] p; // warn
}

// Objective-C, C++
void test(NSUInteger dataLength) {
  int *p = new int;
  NSData *d = [NSData dataWithBytesNoCopy:p
               length:sizeof(int) freeWhenDone:1];
    // warn +dataWithBytesNoCopy:length:freeWhenDone: cannot take
    // ownership of memory allocated by 'new'
}

