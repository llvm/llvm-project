// RUN: %check_clang_tidy %s cppcoreguidelines-owning-memory %t \
// RUN: -config='{CheckOptions: \
// RUN:  {cppcoreguidelines-owning-memory.LegacyResourceProducers: "::malloc;::aligned_alloc;::realloc;::calloc;::fopen;::freopen;::tmpfile", \
// RUN:   cppcoreguidelines-owning-memory.LegacyResourceConsumers: "::free;::realloc;::freopen;::fclose"}}' \
// RUN: -- -nostdlib -nostdinc++

namespace gsl {
template <class T>
using owner = T;
} // namespace gsl

extern "C" {
using size_t = decltype(sizeof(void*));
using FILE = int;

void *malloc(size_t ByteCount);
void *aligned_alloc(size_t Alignment, size_t Size);
void *calloc(size_t Count, size_t SizeSingle);
void *realloc(void *Resource, size_t NewByteCount);
void free(void *Resource);

FILE *tmpfile(void);
FILE *fopen(const char *filename, const char *mode);
FILE *freopen(const char *filename, const char *mode, FILE *stream);
void fclose(FILE *Resource);
}

namespace std {
using ::FILE;
using ::size_t;

using ::fclose;
using ::fopen;
using ::freopen;
using ::tmpfile;

using ::aligned_alloc;
using ::calloc;
using ::free;
using ::malloc;
using ::realloc;
} // namespace std

void nonOwningCall(int *Resource, size_t Size) {}
void nonOwningCall(FILE *Resource) {}

void consumesResource(gsl::owner<int *> Resource, size_t Size) {}
void consumesResource(gsl::owner<FILE *> Resource) {}

void testNonCasted(void *Resource) {}

void testNonCastedOwner(gsl::owner<void *> Resource) {}

FILE *fileFactory1() { return ::fopen("new_file.txt", "w"); }
// CHECK-MESSAGES: [[@LINE-1]]:24: warning: returning a newly created resource of type 'FILE *' (aka 'int *') or 'gsl::owner<>' from a function whose return type is not 'gsl::owner<>'
gsl::owner<FILE *> fileFactory2() { return std::fopen("new_file.txt", "w"); } // Ok

int *arrayFactory1() { return (int *)std::malloc(100); }
// CHECK-MESSAGES: [[@LINE-1]]:24: warning: returning a newly created resource of type 'int *' or 'gsl::owner<>' from a function whose return type is not 'gsl::owner<>'
gsl::owner<int *> arrayFactory2() { return (int *)std::malloc(100); } // Ok
void *dataFactory1() { return std::malloc(100); }
// CHECK-MESSAGES: [[@LINE-1]]:24: warning: returning a newly created resource of type 'void *' or 'gsl::owner<>' from a function whose return type is not 'gsl::owner<>'
gsl::owner<void *> dataFactory2() { return std::malloc(100); } // Ok

void test_resource_creators() {
  const unsigned int ByteCount = 25 * sizeof(int);
  int Bad = 42;

  int *IntArray1 = (int *)std::malloc(ByteCount);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'int *' with a newly created 'gsl::owner<>'
  int *IntArray2 = static_cast<int *>(std::malloc(ByteCount)); // Bad
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'int *' with a newly created 'gsl::owner<>'
  void *IntArray3 = std::malloc(ByteCount);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'void *' with a newly created 'gsl::owner<>'

  int *IntArray4 = (int *)::malloc(ByteCount);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'int *' with a newly created 'gsl::owner<>'
  int *IntArray5 = static_cast<int *>(::malloc(ByteCount)); // Bad
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'int *' with a newly created 'gsl::owner<>'
  void *IntArray6 = ::malloc(ByteCount);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'void *' with a newly created 'gsl::owner<>'

  gsl::owner<int *> IntArray7 = (int *)malloc(ByteCount); // Ok
  gsl::owner<void *> IntArray8 = malloc(ByteCount);       // Ok

  gsl::owner<int *> IntArray9 = &Bad;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: expected initialization with value of type 'gsl::owner<>'; got 'int *'

  nonOwningCall((int *)malloc(ByteCount), 25);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: initializing non-owner argument of type 'int *' with a newly created 'gsl::owner<>'
  nonOwningCall((int *)::malloc(ByteCount), 25);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: initializing non-owner argument of type 'int *' with a newly created 'gsl::owner<>'

  consumesResource((int *)malloc(ByteCount), 25);   // Ok
  consumesResource((int *)::malloc(ByteCount), 25); // Ok

  testNonCasted(malloc(ByteCount));
  // CHECK-MESSAGES: [[@LINE-1]]:17: warning: initializing non-owner argument of type 'void *' with a newly created 'gsl::owner<>'
  testNonCastedOwner(gsl::owner<void *>(malloc(ByteCount))); // Ok
  testNonCastedOwner(malloc(ByteCount));                     // Ok

  FILE *File1 = std::fopen("test_name.txt", "w+");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'FILE *' (aka 'int *') with a newly created 'gsl::owner<>'
  FILE *File2 = ::fopen("test_name.txt", "w+");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'FILE *' (aka 'int *') with a newly created 'gsl::owner<>'

  gsl::owner<FILE *> File3 = ::fopen("test_name.txt", "w+"); // Ok

  FILE *File4;
  File4 = ::fopen("test_name.txt", "w+");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: assigning newly created 'gsl::owner<>' to non-owner 'FILE *' (aka 'int *')

  gsl::owner<FILE *> File5;
  File5 = ::fopen("test_name.txt", "w+"); // Ok
  File5 = File1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: expected assignment source to be of type 'gsl::owner<>'; got 'FILE *' (aka 'int *')

  gsl::owner<FILE *> File6 = File1;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: expected initialization with value of type 'gsl::owner<>'; got 'FILE *' (aka 'int *')

  FILE *File7 = tmpfile();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'FILE *' (aka 'int *') with a newly created 'gsl::owner<>'
  gsl::owner<FILE *> File8 = tmpfile(); // Ok

  nonOwningCall(::fopen("test_name.txt", "r"));
  // CHECK-MESSAGES: [[@LINE-1]]:17: warning: initializing non-owner argument of type 'FILE *' (aka 'int *') with a newly created 'gsl::owner<>'
  nonOwningCall(std::fopen("test_name.txt", "r"));
  // CHECK-MESSAGES: [[@LINE-1]]:17: warning: initializing non-owner argument of type 'FILE *' (aka 'int *') with a newly created 'gsl::owner<>'

  consumesResource(::fopen("test_name.txt", "r")); // Ok

  int *HeapPointer3 = (int *)aligned_alloc(16ul, 4ul * 32ul);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'int *' with a newly created 'gsl::owner<>'
  gsl::owner<int *> HeapPointer4 = static_cast<int *>(aligned_alloc(16ul, 4ul * 32ul)); // Ok

  void *HeapPointer5 = calloc(10ul, 4ul);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'void *' with a newly created 'gsl::owner<>'
  gsl::owner<void *> HeapPointer6 = calloc(10ul, 4ul); // Ok
}

void test_legacy_consumers() {
  int StackInteger = 42;

  int *StackPointer = &StackInteger;
  int *HeapPointer1 = (int *)malloc(100);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'int *' with a newly created 'gsl::owner<>'
  gsl::owner<int *> HeapPointer2 = (int *)malloc(100);

  std::free(StackPointer);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: calling legacy resource function without passing a 'gsl::owner<>'
  std::free(HeapPointer1);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: calling legacy resource function without passing a 'gsl::owner<>'
  std::free(HeapPointer2); // Ok
  // CHECK MESSAGES: [[@LINE-1]]:3: warning: calling legacy resource function without passing a 'gsl::owner<>'

  // FIXME: the check complains about initialization of 'void *' with new created owner.
  // This happens, because the argument of `free` is not marked as 'owner<>' (and cannot be),
  // and the check will not figure out could be meant as owner.
  // This property will probably never be fixed, because it is probably a rather rare
  // use-case and 'owner<>' should be wrapped in RAII classes anyway!
  std::free(std::malloc(100)); // Ok but silly :)
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: initializing non-owner argument of type 'void *' with a newly created 'gsl::owner<>'

  // Demonstrate, that multi-argument functions are diagnosed as well.
  std::realloc(StackPointer, 200);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: calling legacy resource function without passing a 'gsl::owner<>'
  std::realloc(HeapPointer1, 200);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: calling legacy resource function without passing a 'gsl::owner<>'
  std::realloc(HeapPointer2, 200);     // Ok
  std::realloc(std::malloc(100), 200); // Ok but silly
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: initializing non-owner argument of type 'void *' with a newly created 'gsl::owner<>'

  fclose(fileFactory1());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: calling legacy resource function without passing a 'gsl::owner<>'
  fclose(fileFactory2()); // Ok, same as FIXME with `free(malloc(100))` applies here
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: initializing non-owner argument of type 'FILE *' (aka 'int *') with a newly created 'gsl::owner<>'

  gsl::owner<FILE *> File1 = fopen("testfile.txt", "r"); // Ok
  FILE *File2 = freopen("testfile.txt", "w", File1);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'FILE *' (aka 'int *') with a newly created 'gsl::owner<>'
  // CHECK-MESSAGES: [[@LINE-2]]:17: warning: calling legacy resource function without passing a 'gsl::owner<>'
  // FIXME: The warning for not passing and owner<> is a false positive since both the filename and the
  // mode are not supposed to be owners but still pointers. The check is to coarse for
  // this function. Maybe `freopen` gets special treatment.

  gsl::owner<FILE *> File3 = freopen("testfile.txt", "w", File2); // Bad, File2 no owner
  // CHECK-MESSAGES: [[@LINE-1]]:30: warning: calling legacy resource function without passing a 'gsl::owner<>'

  FILE *TmpFile = tmpfile();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'FILE *' (aka 'int *') with a newly created 'gsl::owner<>'
  FILE *File6 = freopen("testfile.txt", "w", TmpFile); // Bad, both return and argument
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: initializing non-owner 'FILE *' (aka 'int *') with a newly created 'gsl::owner<>'
  // CHECK-MESSAGES: [[@LINE-2]]:17: warning: calling legacy resource function without passing a 'gsl::owner<>'
}
