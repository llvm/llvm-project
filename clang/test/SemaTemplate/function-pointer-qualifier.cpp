// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template<class _Ty> inline
	void testparam(_Ty **, _Ty **)
	{
	}

template<class _Ty> inline
	void testparam(_Ty *const *, _Ty **)
	{
	}

template<class _Ty> inline
	void testparam(_Ty **, const _Ty **)
	{
	}

template<class _Ty> inline
	void testparam(_Ty *const *, const _Ty **)
	{
	}

void case0()
{
    void (**p1)();
    void (**p2)();
    testparam(p1, p2);
}
