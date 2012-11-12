// RUN: %clang %s -O0 -emit-llvm -S -o - | FileCheck %s

// This should call rb_define_global_function, not rb_f_chop.
void rb_define_global_function (const char*,void(*)(),int);
static void rb_f_chop();
void Init_String() {
  rb_define_global_function("chop", rb_f_chop, 0);
}
static void rb_f_chop() {
}

// CHECK: call{{.*}}rb_define_global_function

// PR10335
typedef   void (* JSErrorCallback)(void);
void js_GetErrorMessage(void);
void JS_ReportErrorNumber(JSErrorCallback errorCallback, ...);
void Interpret() {
  JS_ReportErrorNumber(js_GetErrorMessage, 0);
  
  // CHECK: call void ({{.*}}, ...)* @JS_ReportErrorNumber({{.*}}@js_GetErrorMessage
}




// PR10337
struct sigaction { int (*_sa_handler)(int); };
typedef int SigHandler ();
typedef struct sigaction sighandler_cxt;
SigHandler *rl_set_sighandler(ohandler)
sighandler_cxt *ohandler;  {
  return 0;
}

void rl_set_signals() {
  SigHandler *oh;
  oh = rl_set_sighandler(0);
}
