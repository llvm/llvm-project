// RUN: %clang_cc1 -ftabstop 3 -fsyntax-only %s 2>&1 | FileCheck -check-prefix=3 -strict-whitespace %s
// RUN: %clang_cc1 -ftabstop 4 -fsyntax-only %s 2>&1 | FileCheck -check-prefix=4 -strict-whitespace %s
// RUN: %clang_cc1 -ftabstop 5 -fsyntax-only %s 2>&1 | FileCheck -check-prefix=5 -strict-whitespace %s

// tab
	void* a = 1;

// tab tab
		void* b = 1;

// 3x space tab
   	void* c = 1;

// tab at column 10
void* d =	1;

//CHECK-3: {{^   void\* a = 1;}}
//CHECK-3: {{^      void\* b = 1;}}
//CHECK-3: {{^      void\* c = 1;}}
//CHECK-3: {{^void\* d =   1;}}

//CHECK-4: {{^    void\* a = 1;}}
//CHECK-4: {{^        void\* b = 1;}}
//CHECK-4: {{^    void\* c = 1;}}
//CHECK-4: {{^void\* d =   1;}}

//CHECK-5: {{^     void\* a = 1;}}
//CHECK-5: {{^          void\* b = 1;}}
//CHECK-5: {{^     void\* c = 1;}}
//CHECK-5: {{^void\* d = 1;}}

// Test code modification hints

void f(void)
{
	if (0	& 1	== 1)
	{}

	if (1 == 0	& 1)
	{}
}

// CHECK-3: {{^   }}if (0 & 1   == 1)
// CHECK-3: {{^   }}        (       )
// CHECK-3: {{^   }}if (1 == 0  & 1)
// CHECK-3: {{^   }}    (     )

// CHECK-4: {{^    }}if (0   & 1 == 1)
// CHECK-4: {{^    }}          (     )
// CHECK-4: {{^    }}if (1 == 0  & 1)
// CHECK-4: {{^    }}    (     )

// CHECK-5: {{^     }}if (0     & 1  == 1)
// CHECK-5: {{^     }}            (      )
// CHECK-5: {{^     }}if (1 == 0     & 1)
// CHECK-5: {{^     }}    (     )
