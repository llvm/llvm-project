// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,unix.API,osx.API,optin.portability %s -analyzer-store=region -analyzer-output=plist -analyzer-eagerly-assume -analyzer-config faux-bodies=true -analyzer-config path-diagnostics-alternate=false -fblocks -verify -o %t.plist
// RUN: FileCheck --input-file=%t.plist %s
// RUN: mkdir -p %t.dir
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.API,osx.API,optin.portability -analyzer-output=html -analyzer-config faux-bodies=true -fblocks -o %t.dir %s
// RUN: rm -fR %t.dir
struct _opaque_pthread_once_t {
  long __sig;
  char __opaque[8];
};
typedef struct _opaque_pthread_once_t    __darwin_pthread_once_t;
typedef __darwin_pthread_once_t pthread_once_t;
int pthread_once(pthread_once_t *, void (*)(void));
typedef long unsigned int __darwin_size_t;
typedef __darwin_size_t size_t;
void *calloc(size_t, size_t);
void *malloc(size_t);
void *realloc(void *, size_t);
void *reallocf(void *, size_t);
void *alloca(size_t);
void *valloc(size_t);
typedef union {
 struct _os_object_s *_os_obj;
 struct dispatch_object_s *_do;
 struct dispatch_continuation_s *_dc;
 struct dispatch_queue_s *_dq;
 struct dispatch_queue_attr_s *_dqa;
 struct dispatch_group_s *_dg;
 struct dispatch_source_s *_ds;
 struct dispatch_source_attr_s *_dsa;
 struct dispatch_semaphore_s *_dsema;
 struct dispatch_data_s *_ddata;
 struct dispatch_io_s *_dchannel;
 struct dispatch_operation_s *_doperation;
 struct dispatch_disk_s *_ddisk;
} dispatch_object_t __attribute__((__transparent_union__));
typedef void (^dispatch_block_t)(void);
typedef long dispatch_once_t;
typedef struct dispatch_queue_s *dispatch_queue_t;
void dispatch_once(dispatch_once_t *predicate, dispatch_block_t block);
void dispatch_sync(dispatch_queue_t queue, dispatch_block_t block);

#ifndef O_CREAT
#define O_CREAT 0x0200
#define O_RDONLY 0x0000
#endif
int open(const char *, int, ...);
int openat(int, const char *, int, ...);
int close(int fildes);

void test_open(const char *path) {
  int fd;
  fd = open(path, O_RDONLY); // no-warning
  if (!fd)
    close(fd);

  fd = open(path, O_CREAT); // expected-warning{{Call to 'open' requires a 3rd argument when the 'O_CREAT' flag is set}}
  if (!fd)
    close(fd);
} 

void test_open_at(int directory_fd, const char *relative_path) {
  int fd;
  fd = openat(directory_fd, relative_path, O_RDONLY); // no-warning
  if (!fd)
    close(fd);

  fd = openat(directory_fd, relative_path, O_CREAT); // expected-warning{{Call to 'openat' requires a 4th argument when the 'O_CREAT' flag is set}}
  if (!fd)
    close(fd);
}

void test_dispatch_once() {
  dispatch_once_t pred = 0;
  do { if (__builtin_expect(*(&pred), ~0l) != ~0l) dispatch_once((&pred), (^() {})); } while (0); // expected-warning{{Call to 'dispatch_once' uses the local variable 'pred' for the predicate value}}
}
void test_dispatch_once_neg() {
  static dispatch_once_t pred = 0;
  do { if (__builtin_expect(*(&pred), ~0l) != ~0l) dispatch_once((&pred), (^() {})); } while (0); // no-warning
}

void test_pthread_once_aux();

void test_pthread_once() {
  pthread_once_t pred = {0x30B1BCBA, {0}};
  pthread_once(&pred, test_pthread_once_aux); // expected-warning{{Call to 'pthread_once' uses the local variable 'pred' for the "control" value}}
}
void test_pthread_once_neg() {
  static pthread_once_t pred = {0x30B1BCBA, {0}};
  pthread_once(&pred, test_pthread_once_aux); // no-warning
}

// PR 2899 - warn of zero-sized allocations to malloc().
void pr2899() {
  char* foo = malloc(0); // expected-warning{{Call to 'malloc' has an allocation size of 0 bytes}}
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void pr2899_nowarn(size_t size) {
  char* foo = malloc(size); // no-warning
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_calloc(void) {
  char *foo = calloc(0, 42); // expected-warning{{Call to 'calloc' has an allocation size of 0 bytes}}
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_calloc2(void) {
  char *foo = calloc(42, 0); // expected-warning{{Call to 'calloc' has an allocation size of 0 bytes}}
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_calloc_nowarn(size_t nmemb, size_t size) {
  char *foo = calloc(nmemb, size); // no-warning
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_realloc(char *ptr) {
  char *foo = realloc(ptr, 0); // expected-warning{{Call to 'realloc' has an allocation size of 0 bytes}}
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_reallocf(char *ptr) {
  char *foo = reallocf(ptr, 0); // expected-warning{{Call to 'reallocf' has an allocation size of 0 bytes}}
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_realloc_nowarn(char *ptr, size_t size) {
  char *foo = realloc(ptr, size); // no-warning
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_reallocf_nowarn(char *ptr, size_t size) {
  char *foo = reallocf(ptr, size); // no-warning
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_alloca() {
  char *foo = alloca(0); // expected-warning{{Call to 'alloca' has an allocation size of 0 bytes}}
  for(unsigned i = 0; i < 100; i++) {
    foo[i] = 0; 
  }
}
void test_alloca_nowarn(size_t sz) {
  char *foo = alloca(sz); // no-warning
  for(unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_builtin_alloca() {
  char *foo2 = __builtin_alloca(0); // expected-warning{{Call to 'alloca' has an allocation size of 0 bytes}}
  for(unsigned i = 0; i < 100; i++) {
    foo2[i] = 0; 
  }
}
void test_builtin_alloca_nowarn(size_t sz) {
  char *foo2 = __builtin_alloca(sz); // no-warning
  for(unsigned i = 0; i < 100; i++) {
    foo2[i] = 0;
  }
}
void test_valloc() {
  char *foo = valloc(0); // expected-warning{{Call to 'valloc' has an allocation size of 0 bytes}}
  for(unsigned i = 0; i < 100; i++) {
    foo[i] = 0; 
  }
}
void test_valloc_nowarn(size_t sz) {
  char *foo = valloc(sz); // no-warning
  for(unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}

// Test dispatch_once being a macro that wraps a call to _dispatch_once, which in turn
// calls the real dispatch_once.

static inline void _dispatch_once(dispatch_once_t *predicate, dispatch_block_t block)
{
  dispatch_once(predicate, block);
}

#define dispatch_once _dispatch_once

void test_dispatch_once_in_macro() {
  dispatch_once_t pred = 0;
  dispatch_once(&pred, ^(){});  // expected-warning {{Call to 'dispatch_once' uses the local variable 'pred' for the predicate value}}
}

// Test inlining of dispatch_sync.
void test_dispatch_sync(dispatch_queue_t queue, int *q) {
  int *p = 0;
  dispatch_sync(queue, ^(void){ 
	  if (q) {
		*p = 1; // expected-warning {{null pointer}}
	   }
  });
}

// Test inlining if dispatch_once.
void test_inline_dispatch_once() {
  static dispatch_once_t pred;
  int *p = 0;
  dispatch_once(&pred, ^(void) {
	  *p = 1; // expected-warning {{null}}
  });
}

// CHECK: <key>diagnostics</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>51</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>51</integer>
// CHECK-NEXT:           <key>col</key><integer>5</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>53</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>53</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>53</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>53</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>53</integer>
// CHECK-NEXT:           <key>col</key><integer>7</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>53</integer>
// CHECK-NEXT:           <key>col</key><integer>7</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>53</integer>
// CHECK-NEXT:      <key>col</key><integer>7</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>53</integer>
// CHECK-NEXT:         <key>col</key><integer>7</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>53</integer>
// CHECK-NEXT:         <key>col</key><integer>9</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Assuming &apos;fd&apos; is not equal to 0</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Assuming &apos;fd&apos; is not equal to 0</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>53</integer>
// CHECK-NEXT:           <key>col</key><integer>7</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>53</integer>
// CHECK-NEXT:           <key>col</key><integer>7</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>56</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>56</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>56</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>56</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>56</integer>
// CHECK-NEXT:           <key>col</key><integer>8</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>56</integer>
// CHECK-NEXT:           <key>col</key><integer>11</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>56</integer>
// CHECK-NEXT:      <key>col</key><integer>8</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>56</integer>
// CHECK-NEXT:         <key>col</key><integer>19</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>56</integer>
// CHECK-NEXT:         <key>col</key><integer>25</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;open&apos; requires a 3rd argument when the &apos;O_CREAT&apos; flag is set</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;open&apos; requires a 3rd argument when the &apos;O_CREAT&apos; flag is set</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;open&apos; requires a 3rd argument when the &apos;O_CREAT&apos; flag is set</string>
// CHECK-NEXT:   <key>category</key><string>Unix API</string>
// CHECK-NEXT:   <key>type</key><string>Improper use of &apos;open&apos;</string>
// CHECK-NEXT:   <key>check_name</key><string>unix.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>5ef25c5c47db3dbfb329901d426452e7</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>test_open</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>6</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>56</integer>
// CHECK-NEXT:   <key>col</key><integer>8</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>62</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>62</integer>
// CHECK-NEXT:           <key>col</key><integer>5</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>64</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>64</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>64</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>64</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>64</integer>
// CHECK-NEXT:           <key>col</key><integer>7</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>64</integer>
// CHECK-NEXT:           <key>col</key><integer>7</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>64</integer>
// CHECK-NEXT:      <key>col</key><integer>7</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>64</integer>
// CHECK-NEXT:         <key>col</key><integer>7</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>64</integer>
// CHECK-NEXT:         <key>col</key><integer>9</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Assuming &apos;fd&apos; is not equal to 0</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Assuming &apos;fd&apos; is not equal to 0</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>64</integer>
// CHECK-NEXT:           <key>col</key><integer>7</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>64</integer>
// CHECK-NEXT:           <key>col</key><integer>7</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>67</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>67</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>67</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>67</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>67</integer>
// CHECK-NEXT:           <key>col</key><integer>8</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>67</integer>
// CHECK-NEXT:           <key>col</key><integer>13</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>67</integer>
// CHECK-NEXT:      <key>col</key><integer>8</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>67</integer>
// CHECK-NEXT:         <key>col</key><integer>44</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>67</integer>
// CHECK-NEXT:         <key>col</key><integer>50</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;openat&apos; requires a 4th argument when the &apos;O_CREAT&apos; flag is set</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;openat&apos; requires a 4th argument when the &apos;O_CREAT&apos; flag is set</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;openat&apos; requires a 4th argument when the &apos;O_CREAT&apos; flag is set</string>
// CHECK-NEXT:   <key>category</key><string>Unix API</string>
// CHECK-NEXT:   <key>type</key><string>Improper use of &apos;open&apos;</string>
// CHECK-NEXT:   <key>check_name</key><string>unix.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>433b17840bb989e7a13dc6337c1ed220</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>test_open_at</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>6</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>67</integer>
// CHECK-NEXT:   <key>col</key><integer>8</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>73</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>73</integer>
// CHECK-NEXT:           <key>col</key><integer>17</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>74</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>74</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>74</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>74</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>74</integer>
// CHECK-NEXT:           <key>col</key><integer>8</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>74</integer>
// CHECK-NEXT:           <key>col</key><integer>9</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>74</integer>
// CHECK-NEXT:           <key>col</key><integer>8</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>74</integer>
// CHECK-NEXT:           <key>col</key><integer>9</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>74</integer>
// CHECK-NEXT:           <key>col</key><integer>52</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>74</integer>
// CHECK-NEXT:           <key>col</key><integer>64</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>74</integer>
// CHECK-NEXT:      <key>col</key><integer>52</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>74</integer>
// CHECK-NEXT:         <key>col</key><integer>66</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>74</integer>
// CHECK-NEXT:         <key>col</key><integer>72</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;dispatch_once&apos; uses the local variable &apos;pred&apos; for the predicate value.  Using such transient memory for the predicate is potentially dangerous.  Perhaps you intended to declare the variable as &apos;static&apos;?</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;dispatch_once&apos; uses the local variable &apos;pred&apos; for the predicate value.  Using such transient memory for the predicate is potentially dangerous.  Perhaps you intended to declare the variable as &apos;static&apos;?</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;dispatch_once&apos; uses the local variable &apos;pred&apos; for the predicate value.  Using such transient memory for the predicate is potentially dangerous.  Perhaps you intended to declare the variable as &apos;static&apos;?</string>
// CHECK-NEXT:   <key>category</key><string>API Misuse (Apple)</string>
// CHECK-NEXT:   <key>type</key><string>Improper use of &apos;dispatch_once&apos;</string>
// CHECK-NEXT:   <key>check_name</key><string>osx.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>173fbcac3fc64dbaec32768d4cfda250</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>test_dispatch_once</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>2</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>74</integer>
// CHECK-NEXT:   <key>col</key><integer>52</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>84</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>84</integer>
// CHECK-NEXT:           <key>col</key><integer>16</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>85</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>85</integer>
// CHECK-NEXT:           <key>col</key><integer>14</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>85</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>85</integer>
// CHECK-NEXT:         <key>col</key><integer>16</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>85</integer>
// CHECK-NEXT:         <key>col</key><integer>20</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;pthread_once&apos; uses the local variable &apos;pred&apos; for the &quot;control&quot; value.  Using such transient memory for the control value is potentially dangerous.  Perhaps you intended to declare the variable as &apos;static&apos;?</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;pthread_once&apos; uses the local variable &apos;pred&apos; for the &quot;control&quot; value.  Using such transient memory for the control value is potentially dangerous.  Perhaps you intended to declare the variable as &apos;static&apos;?</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;pthread_once&apos; uses the local variable &apos;pred&apos; for the &quot;control&quot; value.  Using such transient memory for the control value is potentially dangerous.  Perhaps you intended to declare the variable as &apos;static&apos;?</string>
// CHECK-NEXT:   <key>category</key><string>Unix API</string>
// CHECK-NEXT:   <key>type</key><string>Improper use of &apos;pthread_once&apos;</string>
// CHECK-NEXT:   <key>check_name</key><string>unix.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>1e2f657a65cf8ee3c3ac2227de05ade4</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>test_pthread_once</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>2</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>85</integer>
// CHECK-NEXT:   <key>col</key><integer>3</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>94</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>94</integer>
// CHECK-NEXT:           <key>col</key><integer>6</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>94</integer>
// CHECK-NEXT:           <key>col</key><integer>15</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>94</integer>
// CHECK-NEXT:           <key>col</key><integer>20</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>94</integer>
// CHECK-NEXT:      <key>col</key><integer>15</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>94</integer>
// CHECK-NEXT:         <key>col</key><integer>22</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>94</integer>
// CHECK-NEXT:         <key>col</key><integer>22</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;malloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;malloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;malloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:   <key>category</key><string>Unix API</string>
// CHECK-NEXT:   <key>type</key><string>Undefined allocation of 0 bytes (CERT MEM04-C; CWE-131)</string>
// CHECK-NEXT:   <key>check_name</key><string>unix.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>0e841458f0cb7cf161d35f9db5862dcf</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>pr2899</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>94</integer>
// CHECK-NEXT:   <key>col</key><integer>15</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>106</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>106</integer>
// CHECK-NEXT:           <key>col</key><integer>6</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>106</integer>
// CHECK-NEXT:           <key>col</key><integer>15</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>106</integer>
// CHECK-NEXT:           <key>col</key><integer>20</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>106</integer>
// CHECK-NEXT:      <key>col</key><integer>15</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>106</integer>
// CHECK-NEXT:         <key>col</key><integer>22</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>106</integer>
// CHECK-NEXT:         <key>col</key><integer>22</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;calloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;calloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;calloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:   <key>category</key><string>Unix API</string>
// CHECK-NEXT:   <key>type</key><string>Undefined allocation of 0 bytes (CERT MEM04-C; CWE-131)</string>
// CHECK-NEXT:   <key>check_name</key><string>unix.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>a267ff573c7e8b959a3f886677893eb0</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>test_calloc</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>106</integer>
// CHECK-NEXT:   <key>col</key><integer>15</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>112</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>112</integer>
// CHECK-NEXT:           <key>col</key><integer>6</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>112</integer>
// CHECK-NEXT:           <key>col</key><integer>15</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>112</integer>
// CHECK-NEXT:           <key>col</key><integer>20</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>112</integer>
// CHECK-NEXT:      <key>col</key><integer>15</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>112</integer>
// CHECK-NEXT:         <key>col</key><integer>26</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>112</integer>
// CHECK-NEXT:         <key>col</key><integer>26</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;calloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;calloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;calloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:   <key>category</key><string>Unix API</string>
// CHECK-NEXT:   <key>type</key><string>Undefined allocation of 0 bytes (CERT MEM04-C; CWE-131)</string>
// CHECK-NEXT:   <key>check_name</key><string>unix.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>14eb72957baab3c63bac610a10e6f48b</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>test_calloc2</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>112</integer>
// CHECK-NEXT:   <key>col</key><integer>15</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>124</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>124</integer>
// CHECK-NEXT:           <key>col</key><integer>6</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>124</integer>
// CHECK-NEXT:           <key>col</key><integer>15</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>124</integer>
// CHECK-NEXT:           <key>col</key><integer>21</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>124</integer>
// CHECK-NEXT:      <key>col</key><integer>15</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>124</integer>
// CHECK-NEXT:         <key>col</key><integer>28</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>124</integer>
// CHECK-NEXT:         <key>col</key><integer>28</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;realloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;realloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;realloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:   <key>category</key><string>Unix API</string>
// CHECK-NEXT:   <key>type</key><string>Undefined allocation of 0 bytes (CERT MEM04-C; CWE-131)</string>
// CHECK-NEXT:   <key>check_name</key><string>unix.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>7f6f67ebe3d481aed7750005bea7e371</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>test_realloc</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>124</integer>
// CHECK-NEXT:   <key>col</key><integer>15</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>130</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>130</integer>
// CHECK-NEXT:           <key>col</key><integer>6</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>130</integer>
// CHECK-NEXT:           <key>col</key><integer>15</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>130</integer>
// CHECK-NEXT:           <key>col</key><integer>22</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>130</integer>
// CHECK-NEXT:      <key>col</key><integer>15</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>130</integer>
// CHECK-NEXT:         <key>col</key><integer>29</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>130</integer>
// CHECK-NEXT:         <key>col</key><integer>29</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;reallocf&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;reallocf&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;reallocf&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:   <key>category</key><string>Unix API</string>
// CHECK-NEXT:   <key>type</key><string>Undefined allocation of 0 bytes (CERT MEM04-C; CWE-131)</string>
// CHECK-NEXT:   <key>check_name</key><string>unix.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>4941698efbd81601653dff10ef9c645b</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>test_reallocf</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>130</integer>
// CHECK-NEXT:   <key>col</key><integer>15</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>148</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>148</integer>
// CHECK-NEXT:           <key>col</key><integer>6</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>148</integer>
// CHECK-NEXT:           <key>col</key><integer>15</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>148</integer>
// CHECK-NEXT:           <key>col</key><integer>20</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>148</integer>
// CHECK-NEXT:      <key>col</key><integer>15</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>148</integer>
// CHECK-NEXT:         <key>col</key><integer>22</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>148</integer>
// CHECK-NEXT:         <key>col</key><integer>22</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;alloca&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;alloca&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;alloca&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:   <key>category</key><string>Unix API</string>
// CHECK-NEXT:   <key>type</key><string>Undefined allocation of 0 bytes (CERT MEM04-C; CWE-131)</string>
// CHECK-NEXT:   <key>check_name</key><string>unix.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>b7ca3488e81d9d9d4b8dc545258ce97c</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>test_alloca</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>148</integer>
// CHECK-NEXT:   <key>col</key><integer>15</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>160</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>160</integer>
// CHECK-NEXT:           <key>col</key><integer>6</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>160</integer>
// CHECK-NEXT:           <key>col</key><integer>16</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>160</integer>
// CHECK-NEXT:           <key>col</key><integer>31</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>160</integer>
// CHECK-NEXT:      <key>col</key><integer>16</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>160</integer>
// CHECK-NEXT:         <key>col</key><integer>33</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>160</integer>
// CHECK-NEXT:         <key>col</key><integer>33</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;alloca&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;alloca&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;alloca&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:   <key>category</key><string>Unix API</string>
// CHECK-NEXT:   <key>type</key><string>Undefined allocation of 0 bytes (CERT MEM04-C; CWE-131)</string>
// CHECK-NEXT:   <key>check_name</key><string>unix.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>1ec52551362b070237f47f6bb6c3847d</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>test_builtin_alloca</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>160</integer>
// CHECK-NEXT:   <key>col</key><integer>16</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>172</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>172</integer>
// CHECK-NEXT:           <key>col</key><integer>6</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>172</integer>
// CHECK-NEXT:           <key>col</key><integer>15</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>172</integer>
// CHECK-NEXT:           <key>col</key><integer>20</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>172</integer>
// CHECK-NEXT:      <key>col</key><integer>15</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>172</integer>
// CHECK-NEXT:         <key>col</key><integer>22</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>172</integer>
// CHECK-NEXT:         <key>col</key><integer>22</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;valloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;valloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;valloc&apos; has an allocation size of 0 bytes</string>
// CHECK-NEXT:   <key>category</key><string>Unix API</string>
// CHECK-NEXT:   <key>type</key><string>Undefined allocation of 0 bytes (CERT MEM04-C; CWE-131)</string>
// CHECK-NEXT:   <key>check_name</key><string>unix.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>675741e04c8d0071d280324e23f41d35</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>test_valloc</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>172</integer>
// CHECK-NEXT:   <key>col</key><integer>15</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>195</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>195</integer>
// CHECK-NEXT:           <key>col</key><integer>17</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>196</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>196</integer>
// CHECK-NEXT:           <key>col</key><integer>15</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>196</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>196</integer>
// CHECK-NEXT:         <key>col</key><integer>17</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>196</integer>
// CHECK-NEXT:         <key>col</key><integer>21</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Call to &apos;dispatch_once&apos; uses the local variable &apos;pred&apos; for the predicate value.  Using such transient memory for the predicate is potentially dangerous.  Perhaps you intended to declare the variable as &apos;static&apos;?</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Call to &apos;dispatch_once&apos; uses the local variable &apos;pred&apos; for the predicate value.  Using such transient memory for the predicate is potentially dangerous.  Perhaps you intended to declare the variable as &apos;static&apos;?</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Call to &apos;dispatch_once&apos; uses the local variable &apos;pred&apos; for the predicate value.  Using such transient memory for the predicate is potentially dangerous.  Perhaps you intended to declare the variable as &apos;static&apos;?</string>
// CHECK-NEXT:   <key>category</key><string>API Misuse (Apple)</string>
// CHECK-NEXT:   <key>type</key><string>Improper use of &apos;dispatch_once&apos;</string>
// CHECK-NEXT:   <key>check_name</key><string>osx.API</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>8ded1f2025c1e4a4bcd5302dc97006d9</string>
// CHECK-NEXT:  <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:  <key>issue_context</key><string>test_dispatch_once_in_macro</string>
// CHECK-NEXT:  <key>issue_hash_function_offset</key><string>2</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>196</integer>
// CHECK-NEXT:   <key>col</key><integer>3</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>201</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>201</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>201</integer>
// CHECK-NEXT:         <key>col</key><integer>8</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>&apos;p&apos; initialized to a null pointer value</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>&apos;p&apos; initialized to a null pointer value</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>201</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>201</integer>
// CHECK-NEXT:           <key>col</key><integer>5</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>202</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>202</integer>
// CHECK-NEXT:           <key>col</key><integer>15</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>202</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>202</integer>
// CHECK-NEXT:           <key>col</key><integer>15</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>202</integer>
// CHECK-NEXT:           <key>col</key><integer>24</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>202</integer>
// CHECK-NEXT:           <key>col</key><integer>24</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>202</integer>
// CHECK-NEXT:      <key>col</key><integer>24</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>202</integer>
// CHECK-NEXT:         <key>col</key><integer>24</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>206</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>&apos;p&apos; captured by block as a null pointer value</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>&apos;p&apos; captured by block as a null pointer value</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>202</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>202</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>206</integer>
// CHECK-NEXT:         <key>col</key><integer>4</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Calling &apos;dispatch_sync&apos;</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Calling &apos;dispatch_sync&apos;</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>202</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>202</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>206</integer>
// CHECK-NEXT:         <key>col</key><integer>4</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>1</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Calling anonymous block</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Calling anonymous block</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>202</integer>
// CHECK-NEXT:      <key>col</key><integer>24</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>depth</key><integer>2</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Entered call from &apos;dispatch_sync&apos;</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Entered call from &apos;dispatch_sync&apos;</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>202</integer>
// CHECK-NEXT:           <key>col</key><integer>24</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>202</integer>
// CHECK-NEXT:           <key>col</key><integer>24</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>203</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>203</integer>
// CHECK-NEXT:           <key>col</key><integer>5</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>203</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>203</integer>
// CHECK-NEXT:           <key>col</key><integer>5</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>203</integer>
// CHECK-NEXT:           <key>col</key><integer>8</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>203</integer>
// CHECK-NEXT:           <key>col</key><integer>8</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>203</integer>
// CHECK-NEXT:      <key>col</key><integer>8</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>203</integer>
// CHECK-NEXT:         <key>col</key><integer>8</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>203</integer>
// CHECK-NEXT:         <key>col</key><integer>8</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>2</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Assuming &apos;q&apos; is non-null</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Assuming &apos;q&apos; is non-null</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>203</integer>
// CHECK-NEXT:           <key>col</key><integer>8</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>203</integer>
// CHECK-NEXT:           <key>col</key><integer>8</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>204</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>204</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>204</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>204</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>204</integer>
// CHECK-NEXT:           <key>col</key><integer>6</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>204</integer>
// CHECK-NEXT:           <key>col</key><integer>6</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>204</integer>
// CHECK-NEXT:      <key>col</key><integer>6</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>204</integer>
// CHECK-NEXT:         <key>col</key><integer>4</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>204</integer>
// CHECK-NEXT:         <key>col</key><integer>4</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>2</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK-NEXT:   <key>category</key><string>Logic error</string>
// CHECK-NEXT:   <key>type</key><string>Dereference of null pointer</string>
// CHECK-NEXT:   <key>check_name</key><string>core.NullDereference</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>5d3f4c433004c7a6d4a06aa30cc3ea85</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>204</integer>
// CHECK-NEXT:   <key>col</key><integer>6</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>path</key>
// CHECK-NEXT:   <array>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>211</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>211</integer>
// CHECK-NEXT:           <key>col</key><integer>8</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>212</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>212</integer>
// CHECK-NEXT:           <key>col</key><integer>5</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>212</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>212</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>212</integer>
// CHECK-NEXT:         <key>col</key><integer>8</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>&apos;p&apos; initialized to a null pointer value</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>&apos;p&apos; initialized to a null pointer value</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>212</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>212</integer>
// CHECK-NEXT:           <key>col</key><integer>5</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>213</integer>
// CHECK-NEXT:           <key>col</key><integer>24</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>213</integer>
// CHECK-NEXT:           <key>col</key><integer>24</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>213</integer>
// CHECK-NEXT:      <key>col</key><integer>24</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>213</integer>
// CHECK-NEXT:         <key>col</key><integer>24</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>215</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>&apos;p&apos; captured by block as a null pointer value</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>&apos;p&apos; captured by block as a null pointer value</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>213</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>213</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>215</integer>
// CHECK-NEXT:         <key>col</key><integer>4</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>0</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Calling &apos;_dispatch_once&apos;</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Calling &apos;_dispatch_once&apos;</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>187</integer>
// CHECK-NEXT:      <key>col</key><integer>1</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>depth</key><integer>1</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Entered call from &apos;test_inline_dispatch_once&apos;</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Entered call from &apos;test_inline_dispatch_once&apos;</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>187</integer>
// CHECK-NEXT:           <key>col</key><integer>1</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>187</integer>
// CHECK-NEXT:           <key>col</key><integer>6</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>189</integer>
// CHECK-NEXT:           <key>col</key><integer>3</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>189</integer>
// CHECK-NEXT:           <key>col</key><integer>15</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>189</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>189</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>189</integer>
// CHECK-NEXT:         <key>col</key><integer>33</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>1</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Calling &apos;dispatch_once&apos;</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Calling &apos;dispatch_once&apos;</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>189</integer>
// CHECK-NEXT:      <key>col</key><integer>3</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>189</integer>
// CHECK-NEXT:         <key>col</key><integer>3</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>189</integer>
// CHECK-NEXT:         <key>col</key><integer>33</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>2</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Calling anonymous block</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Calling anonymous block</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>213</integer>
// CHECK-NEXT:      <key>col</key><integer>24</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>depth</key><integer>3</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Entered call from &apos;dispatch_once&apos;</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Entered call from &apos;dispatch_once&apos;</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>213</integer>
// CHECK-NEXT:           <key>col</key><integer>24</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>213</integer>
// CHECK-NEXT:           <key>col</key><integer>24</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>214</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>214</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>control</string>
// CHECK-NEXT:     <key>edges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:       <dict>
// CHECK-NEXT:        <key>start</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>214</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>214</integer>
// CHECK-NEXT:           <key>col</key><integer>4</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:        <key>end</key>
// CHECK-NEXT:         <array>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>214</integer>
// CHECK-NEXT:           <key>col</key><integer>7</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:          <dict>
// CHECK-NEXT:           <key>line</key><integer>214</integer>
// CHECK-NEXT:           <key>col</key><integer>7</integer>
// CHECK-NEXT:           <key>file</key><integer>0</integer>
// CHECK-NEXT:          </dict>
// CHECK-NEXT:         </array>
// CHECK-NEXT:       </dict>
// CHECK-NEXT:      </array>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:    <dict>
// CHECK-NEXT:     <key>kind</key><string>event</string>
// CHECK-NEXT:     <key>location</key>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>line</key><integer>214</integer>
// CHECK-NEXT:      <key>col</key><integer>7</integer>
// CHECK-NEXT:      <key>file</key><integer>0</integer>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <key>ranges</key>
// CHECK-NEXT:     <array>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>214</integer>
// CHECK-NEXT:         <key>col</key><integer>5</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>line</key><integer>214</integer>
// CHECK-NEXT:         <key>col</key><integer>5</integer>
// CHECK-NEXT:         <key>file</key><integer>0</integer>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </array>
// CHECK-NEXT:     <key>depth</key><integer>3</integer>
// CHECK-NEXT:     <key>extended_message</key>
// CHECK-NEXT:     <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK-NEXT:     <key>message</key>
// CHECK-NEXT:     <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK-NEXT:    </dict>
// CHECK-NEXT:   </array>
// CHECK-NEXT:   <key>description</key><string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK-NEXT:   <key>category</key><string>Logic error</string>
// CHECK-NEXT:   <key>type</key><string>Dereference of null pointer</string>
// CHECK-NEXT:   <key>check_name</key><string>core.NullDereference</string>
// CHECK-NEXT:   <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:   <key>issue_hash_content_of_line_in_context</key><string>265c4fd608dafee211bfa93d21c28866</string>
// CHECK-NEXT:  <key>location</key>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>line</key><integer>214</integer>
// CHECK-NEXT:   <key>col</key><integer>7</integer>
// CHECK-NEXT:   <key>file</key><integer>0</integer>
// CHECK-NEXT:  </dict>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>
