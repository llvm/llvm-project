// RUN: %check_clang_tidy %s concurrency-mt-unsafe %t -- -config='{CheckOptions: {concurrency-mt-unsafe.FunctionSet: "posix"}}'

extern unsigned int sleep (unsigned int __seconds);
extern int *gmtime (const int *__timer);
extern int *gmtime_r (const int *__timer, char*);
extern char *dirname (char *__path);

void foo() {
  sleep(2);
  ::sleep(2);

  auto tm = gmtime(nullptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: function is not thread safe [concurrency-mt-unsafe]
  tm = ::gmtime(nullptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: function is not thread safe [concurrency-mt-unsafe]

  tm = gmtime_r(nullptr, nullptr);
  tm = ::gmtime_r(nullptr, nullptr);

  dirname(nullptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function is not thread safe [concurrency-mt-unsafe]
}
