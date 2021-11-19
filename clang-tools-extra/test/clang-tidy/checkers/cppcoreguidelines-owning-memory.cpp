// RUN: %check_clang_tidy %s cppcoreguidelines-owning-memory %t

namespace gsl {
template <class T>
using owner = T;
} // namespace gsl

template <typename T>
class unique_ptr {
public:
  unique_ptr(gsl::owner<T> resource) : memory(resource) {}
  unique_ptr(const unique_ptr<T> &) = default;

  ~unique_ptr() { delete memory; }

private:
  gsl::owner<T> memory;
};

void takes_owner(gsl::owner<int *> owned_int) {
}

void takes_pointer(int *unowned_int) {
}

void takes_owner_and_more(int some_int, gsl::owner<int *> owned_int, float f) {
}

template <typename T>
void takes_templated_owner(gsl::owner<T> owned_T) {
}

gsl::owner<int *> returns_owner1() { return gsl::owner<int *>(new int(42)); } // Ok
gsl::owner<int *> returns_owner2() { return new int(42); }                    // Ok

int *returns_no_owner1() { return nullptr; }
int *returns_no_owner2() {
  return new int(42);
  // CHECK-NOTES: [[@LINE-1]]:3: warning: returning a newly created resource of type 'int *' or 'gsl::owner<>' from a function whose return type is not 'gsl::owner<>'
}
int *returns_no_owner3() {
  int *should_be_owner = new int(42);
  // CHECK-NOTES: [[@LINE-1]]:3: warning: initializing non-owner 'int *' with a newly created 'gsl::owner<>'
  return should_be_owner;
}
int *returns_no_owner4() {
  gsl::owner<int *> owner = new int(42);
  return owner;
  // CHECK-NOTES: [[@LINE-1]]:3: warning: returning a newly created resource of type 'int *' or 'gsl::owner<>' from a function whose return type is not 'gsl::owner<>'
}

unique_ptr<int *> returns_no_owner5() {
  return unique_ptr<int *>(new int(42)); // Ok
}

/// FIXME: CSA finds it, but the report is misleading. Ownersemantics can catch this
/// by flow analysis similar to bugprone-use-after-move.
void csa_not_finding_leak() {
  gsl::owner<int *> o1 = new int(42); // Ok

  gsl::owner<int *> o2 = o1; // Ok
  o2 = new int(45);          // conceptual leak, the memory from o1 is now leaked, since its considered moved in the guidelines

  delete o2;
  // actual leak occurs here, its found, but mixed
  delete o1;
}

void test_assignment_and_initialization() {
  int stack_int1 = 15;
  int stack_int2;

  gsl::owner<int *> owned_int1 = &stack_int1; // BAD
  // CHECK-NOTES: [[@LINE-1]]:3: warning: expected initialization with value of type 'gsl::owner<>'; got 'int *'

  gsl::owner<int *> owned_int2;
  owned_int2 = &stack_int2; // BAD since no owner, bad since uninitialized
  // CHECK-NOTES: [[@LINE-1]]:3: warning: expected assignment source to be of type 'gsl::owner<>'; got 'int *'

  gsl::owner<int *> owned_int3 = new int(42); // Good
  owned_int3 = nullptr;                       // Good

  gsl::owner<int *> owned_int4(nullptr); // Ok
  owned_int4 = new int(42);              // Good

  gsl::owner<int *> owned_int5 = owned_int3; // Good

  gsl::owner<int *> owned_int6{nullptr}; // Ok
  owned_int6 = owned_int4;               // Good

  // FIXME:, flow analysis for the case of reassignment. Value must be released before
  owned_int6 = owned_int3; // BAD, because reassignment without resource release

  auto owned_int7 = returns_owner1(); // Ok, since type deduction does not eliminate the owner wrapper

  const auto owned_int8 = returns_owner2(); // Ok, since type deduction does not eliminate the owner wrapper

  gsl::owner<int *> owned_int9 = returns_owner1(); // Ok
  int *unowned_int3 = returns_owner1();            // Bad
  // CHECK-NOTES: [[@LINE-1]]:3: warning: initializing non-owner 'int *' with a newly created 'gsl::owner<>'

  gsl::owner<int *> owned_int10;
  owned_int10 = returns_owner1(); // Ok

  int *unowned_int4;
  unowned_int4 = returns_owner1(); // Bad
  // CHECK-NOTES: [[@LINE-1]]:3: warning: assigning newly created 'gsl::owner<>' to non-owner 'int *'

  gsl::owner<int *> owned_int11 = returns_no_owner1(); // Bad since no owner
  // CHECK-NOTES: [[@LINE-1]]:3: warning: expected initialization with value of type 'gsl::owner<>'; got 'int *'

  gsl::owner<int *> owned_int12;
  owned_int12 = returns_no_owner1(); // Bad since no owner
  // CHECK-NOTES: [[@LINE-1]]:3: warning: expected assignment source to be of type 'gsl::owner<>'; got 'int *'

  int *unowned_int5 = returns_no_owner1(); // Ok
  int *unowned_int6;
  unowned_int6 = returns_no_owner1(); // Ok

  int *unowned_int7 = new int(42); // Bad, since resource not assigned to an owner
  // CHECK-NOTES: [[@LINE-1]]:3: warning: initializing non-owner 'int *' with a newly created 'gsl::owner<>'

  int *unowned_int8;
  unowned_int8 = new int(42);
  // CHECK-NOTES: [[@LINE-1]]:3: warning: assigning newly created 'gsl::owner<>' to non-owner 'int *'

  gsl::owner<int *> owned_int13 = nullptr; // Ok
}

void test_deletion() {
  gsl::owner<int *> owned_int1 = new int(42);
  delete owned_int1; // Good

  gsl::owner<int *> owned_int2 = new int[42];
  delete[] owned_int2; // Good

  int *unowned_int1 = new int(42); // BAD, since new creates and owner
  // CHECK-NOTES: [[@LINE-1]]:3: warning: initializing non-owner 'int *' with a newly created 'gsl::owner<>'
  delete unowned_int1; // BAD, since no owner
  // CHECK-NOTES: [[@LINE-1]]:3: warning: deleting a pointer through a type that is not marked 'gsl::owner<>'; consider using a smart pointer instead
  // CHECK-NOTES: [[@LINE-4]]:3: note: variable declared here

  int *unowned_int2 = new int[42]; // BAD, since new creates and owner
  // CHECK-NOTES: [[@LINE-1]]:3: warning: initializing non-owner 'int *' with a newly created 'gsl::owner<>'
  delete[] unowned_int2; // BAD since no owner
  // CHECK-NOTES: [[@LINE-1]]:3: warning: deleting a pointer through a type that is not marked 'gsl::owner<>'; consider using a smart pointer instead
  // CHECK-NOTES: [[@LINE-4]]:3: note: variable declared here

  delete new int(42);   // Technically ok, but stupid
  delete[] new int[42]; // Technically ok, but stupid
}

void test_owner_function_calls() {
  int stack_int = 42;
  int *unowned_int1 = &stack_int;
  takes_owner(&stack_int); // BAD
  // CHECK-NOTES: [[@LINE-1]]:15: warning: expected argument of type 'gsl::owner<>'; got 'int *'
  takes_owner(unowned_int1); // BAD
  // CHECK-NOTES: [[@LINE-1]]:15: warning: expected argument of type 'gsl::owner<>'; got 'int *'

  gsl::owner<int *> owned_int1 = new int(42);
  takes_owner(owned_int1); // Ok

  takes_owner_and_more(42, &stack_int, 42.0f); // BAD
  // CHECK-NOTES: [[@LINE-1]]:28: warning: expected argument of type 'gsl::owner<>'; got 'int *'
  takes_owner_and_more(42, unowned_int1, 42.0f); // BAD
  // CHECK-NOTES: [[@LINE-1]]:28: warning: expected argument of type 'gsl::owner<>'; got 'int *'

  takes_owner_and_more(42, new int(42), 42.0f); // Ok, since new is consumed by owner
  takes_owner_and_more(42, owned_int1, 42.0f);  // Ok, since owner as argument

  takes_templated_owner(owned_int1);   // Ok
  takes_templated_owner(new int(42));  // Ok
  takes_templated_owner(unowned_int1); // Bad
  // CHECK-NOTES: [[@LINE-1]]:25: warning: expected argument of type 'gsl::owner<>'; got 'int *'

  takes_owner(returns_owner1());    // Ok
  takes_owner(returns_no_owner1()); // BAD
  // CHECK-NOTES: [[@LINE-1]]:15: warning: expected argument of type 'gsl::owner<>'; got 'int *'
}

void test_unowned_function_calls() {
  int stack_int = 42;
  int *unowned_int1 = &stack_int;
  gsl::owner<int *> owned_int1 = new int(42);

  takes_pointer(&stack_int);   // Ok
  takes_pointer(unowned_int1); // Ok
  takes_pointer(owned_int1);   // Ok
  takes_pointer(new int(42));  // Bad, since new creates and owner
  // CHECK-NOTES: [[@LINE-1]]:17: warning: initializing non-owner argument of type 'int *' with a newly created 'gsl::owner<>'

  takes_pointer(returns_owner1()); // Bad
  // CHECK-NOTES: [[@LINE-1]]:17: warning: initializing non-owner argument of type 'int *' with a newly created 'gsl::owner<>'

  takes_pointer(returns_no_owner1()); // Ok
}

// FIXME: Typedefing owner<> to something else does not work.
// This might be necessary for code already having a similar typedef like owner<> and
// replacing it with owner<>. This might be the same problem as with templates.
// The canonical type will ignore the owner<> alias, since its a typedef as well.
//
// Check, if owners hidden by typedef are handled the same as 'obvious' owners.
#if 0
using heap_int = gsl::owner<int *>;
typedef gsl::owner<float *> heap_float;

// This tests only a subset, assuming that the check will either see through the
// typedef or not (it doesn't!).
void test_typedefed_values() {
  // Modern typedef.
  int StackInt1 = 42;
  heap_int HeapInt1 = &StackInt1;
  // CHECK MESSAGES: [[@LINE-1]]:3: warning: expected assignment source to be of type 'gsl::owner<>'; got 'int *'

  //FIXME: Typedef not considered correctly here.
  // heap_int HeapInt2 = new int(42); // Ok
  takes_pointer(HeapInt1); // Ok
  takes_owner(HeapInt1);   // Ok

  // Traditional typedef.
  float StackFloat1 = 42.0f;
  heap_float HeapFloat1 = &StackFloat1;
  // CHECK MESSAGES: [[@LINE-1]]:3: warning: expected assignment source to be of type 'gsl::owner<>'; got 'float *'

  //FIXME: Typedef not considered correctly here.
  // heap_float HeapFloat2 = new float(42.0f);
  HeapFloat2 = HeapFloat1; // Ok
}
#endif

struct ArbitraryClass {};
struct ClassWithOwner {                    // Does not define destructor, necessary with owner
  ClassWithOwner() : owner_var(nullptr) {} // Ok

  ClassWithOwner(ArbitraryClass &other) : owner_var(&other) {}
  // CHECK-NOTES: [[@LINE-1]]:43: warning: expected initialization of owner member variable with value of type 'gsl::owner<>'; got 'ArbitraryClass *'

  ClassWithOwner(gsl::owner<ArbitraryClass *> other) : owner_var(other) {} // Ok

  ClassWithOwner(gsl::owner<ArbitraryClass *> data, int /* unused */) { // Ok
    owner_var = data;                                                   // Ok
  }

  ClassWithOwner(ArbitraryClass *bad_data, int /* unused */, int /* unused */) {
    owner_var = bad_data;
    // CHECK-NOTES: [[@LINE-1]]:5: warning: expected assignment source to be of type 'gsl::owner<>'; got 'ArbitraryClass *'
  }

  ClassWithOwner(ClassWithOwner &&other) : owner_var{other.owner_var} {} // Ok

  ClassWithOwner &operator=(ClassWithOwner &&other) {
    owner_var = other.owner_var; // Ok
    return *this;
  }

  // Returning means, that the owner is "moved", so the class should not access this
  // variable anymore after this method gets called.
  gsl::owner<ArbitraryClass *> buggy_but_returns_owner() { return owner_var; }

  gsl::owner<ArbitraryClass *> owner_var;
  // CHECK-NOTES: [[@LINE-1]]:3: warning: member variable of type 'gsl::owner<>' requires the class 'ClassWithOwner' to implement a destructor to release the owned resource
};

class DefaultedDestructor {         // Bad since default constructor with owner
  ~DefaultedDestructor() = default; // Bad, since will not destroy the owner
  gsl::owner<int *> Owner;
  // CHECK-NOTES: [[@LINE-1]]:3: warning: member variable of type 'gsl::owner<>' requires the class 'DefaultedDestructor' to implement a destructor to release the owned resource
};

struct DeletedDestructor {
  ~DeletedDestructor() = delete;
  gsl::owner<int *> Owner;
  // CHECK-NOTES: [[@LINE-1]]:3: warning: member variable of type 'gsl::owner<>' requires the class 'DeletedDestructor' to implement a destructor to release the owned resource
};

void test_class_with_owner() {
  ArbitraryClass A;
  ClassWithOwner C1;                                                   // Ok
  ClassWithOwner C2{A};                                                // Bad, since the owner would be initialized with an non-owner, but catched in the class
  ClassWithOwner C3{gsl::owner<ArbitraryClass *>(new ArbitraryClass)}; // Ok

  const auto Owner1 = C3.buggy_but_returns_owner(); // Ok, deduces Owner1 to owner<ArbitraryClass *> const

  auto Owner2 = C2.buggy_but_returns_owner(); // Ok, deduces Owner2 to owner<ArbitraryClass *>

  Owner2 = &A; // BAD, since type deduction resulted in owner<ArbitraryClass *>
  // CHECK-NOTES: [[@LINE-1]]:3: warning: expected assignment source to be of type 'gsl::owner<>'; got 'ArbitraryClass *'

  gsl::owner<ArbitraryClass *> Owner3 = C1.buggy_but_returns_owner(); // Ok, still an owner
  Owner3 = &A;                                                        // Bad, since assignment of non-owner to owner
  // CHECK-NOTES: [[@LINE-1]]:3: warning: expected assignment source to be of type 'gsl::owner<>'; got 'ArbitraryClass *'
}

template <typename T>
struct HeapArray {                                          // Ok, since destructor with owner
  HeapArray() : _data(nullptr), size(0) {}                  // Ok
  HeapArray(int size) : _data(new int[size]), size(size) {} // Ok
  HeapArray(int size, T val) {
    _data = new int[size]; // Ok
    size = size;
    for (auto i = 0u; i < size; ++i)
      _data[i] = val; // Ok
  }
  HeapArray(int size, T val, int *problematic) : _data{problematic}, size(size) {} // Bad
  // CHECK-NOTES: [[@LINE-1]]:50: warning: expected initialization of owner member variable with value of type 'gsl::owner<>'; got 'void'
  // FIXME: void is incorrect type, probably wrong thing matched

  HeapArray(HeapArray &&other) : _data(other._data), size(other.size) { // Ok
    other._data = nullptr;                                              // Ok
    other.size = 0;
  }

  HeapArray<T> &operator=(HeapArray<T> &&other) {
    _data = other._data; // Ok, NOLINT warning here about bad types, why?
    size = other.size;
    return *this;
  }

  ~HeapArray() { delete[] _data; } // Ok

  T *data() { return _data; } // Ok NOLINT, because it "looks" like a factory

  gsl::owner<T *> _data;
  unsigned int size;
};

void test_inner_template() {
  HeapArray<int> Array1;
  HeapArray<int> Array2(100);
  HeapArray<int> Array3(100, 0);
  HeapArray<int> Array4(100, 0, nullptr);

  Array1 = static_cast<HeapArray<int> &&>(Array2);
  HeapArray<int> Array5(static_cast<HeapArray<int> &&>(Array3));

  int *NonOwningPtr = Array1.data();           // Ok
  gsl::owner<int *> OwningPtr = Array1.data(); // Bad, since it does not return the owner
  // CHECK-NOTES: [[@LINE-1]]:3: warning: expected initialization with value of type 'gsl::owner<>'; got 'int *'
}

// FIXME: Typededuction removes the owner - wrapper, therefore gsl::owner can not be used
// with Template classes like this. Is there a walkaround?
template <typename T>
struct TemplateValue {
  TemplateValue() = default;
  TemplateValue(T t) : val{t} {}

  void setVal(const T &t) { val = t; }
  const T getVal() const { return val; }

  T val;
};

// FIXME: Same typededcution problems
template <typename T>
void template_function(T t) {
  gsl::owner<int *> owner_t = t; // Probably bad, since type deduction still wrong
  // CHECK-NOTES: [[@LINE-1]]:3: warning: expected initialization with value of type 'gsl::owner<>'; got 'T'
  // CHECK-NOTES: [[@LINE-2]]:3: warning: expected initialization with value of type 'gsl::owner<>'; got 'int *'
}

// FIXME: Same typededcution problems
void test_templates() {
  int stack_int = 42;
  int *stack_ptr1 = &stack_int;

  TemplateValue<gsl::owner<int *>> Owner0; // Ok, T should be owner, but is int*

  TemplateValue<gsl::owner<int *>> Owner1(new int(42)); // Ok, T should be owner, but is int*
  Owner1.setVal(&stack_int);                            // Bad since non-owner assignment
  Owner1.setVal(stack_ptr1);                            // Bad since non-owner assignment
  //Owner1.setVal(new int(42)); // Ok, but since type deduction is wrong, this one is considered harmful

  int *stack_ptr2 = Owner1.getVal(); // Bad, initializing non-owner with owner

  TemplateValue<int *> NonOwner1(new int(42));      // Bad, T is int *, hence dynamic memory to non-owner
  gsl::owner<int *> IntOwner1 = NonOwner1.getVal(); // Bad, since owner initialized with non-owner
  // CHECK-NOTES: [[@LINE-1]]:3: warning: expected initialization with value of type 'gsl::owner<>'; got 'int *'

  template_function(IntOwner1);  // Ok, but not actually ok, since type deduction removes owner
  template_function(stack_ptr1); // Bad, but type deduction gets it wrong
}
