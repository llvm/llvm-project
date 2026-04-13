// RUN: %clang_cc1 -std=c++11 -verify %s
// expected-no-diagnostics

// Do not crash when __PRETTY_FUNCTION__ appears in the trailing return type of the lambda
void foo() {
	[]() -> decltype(static_cast<const char*>(__PRETTY_FUNCTION__)) {
		return nullptr;
	}();

#ifdef MS
	[]() -> decltype(static_cast<const char*>(__FUNCSIG__)) {
		return nullptr;
	}();
#endif
}
