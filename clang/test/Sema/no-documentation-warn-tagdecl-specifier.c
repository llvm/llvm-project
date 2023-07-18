// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -verify %s
// rdar://12390371

/** @return s Test*/
struct s* f(void);
struct s;

struct s1;
/** @return s1 Test 1*/
struct s1* f1(void);

struct s2;
/** @return s2 Test 2*/
struct s2* f2(void);
struct s2;

// expected-warning@+1 {{'@return' command used in a comment that is not attached to a function or method declaration}}
/** @return s3 Test 3 - expected warning here */
struct s3;
struct s3* f3(void);

/** @return s4 Test 4 */
struct s4* f4(void);
struct s4 { int is; };

// expected-warning@+1 {{'@return' command used in a comment that is not attached to a function or method declaration}}
/** @return s5 Test 5  - expected warning here */
struct s5 { int is; };
struct s5* f5(void);

// expected-warning@+1 {{'@return' command used in a comment that is not attached to a function or method declaration}}
/** @return s6 Test 6  - expected warning here */
struct s6 *ps6;
struct s6* f6(void);

// expected-warning@+1 {{'@return' command used in a comment that is not attached to a function or method declaration}}
/** @return s7 Test 7  - expected warning here */
struct s7;
struct s7* f7(void);

struct s8 { int is8; };
/** @return s8 Test 8 */
struct s4 *f8(struct s8 *p);


/** @return e Test*/
enum e* g(void);
enum e;

enum e1;
/** @return e1 Test 1*/
enum e1* g1(void);

enum e2;
/** @return e2 Test 2*/
enum e2* g2(void);
enum e2;

// expected-warning@+1 {{'@return' command used in a comment that is not attached to a function or method declaration}}
/** @return e3 Test 3 - expected warning here */
enum e3;
enum e3* g3(void);

/** @return e4 Test 4 */
enum e4* g4(void);
enum e4 { one };

// expected-warning@+1 {{'@return' command used in a comment that is not attached to a function or method declaration}}
/** @return e5 Test 5  - expected warning here */
enum e5 { two };
enum e5* g5(void);

// expected-warning@+1 {{'@return' command used in a comment that is not attached to a function or method declaration}}
/** @return e6 Test 6  - expected warning here */
enum e6 *pe6;
enum e6* g6(void);

// expected-warning@+1 {{'@return' command used in a comment that is not attached to a function or method declaration}}
/** @return e7 Test 7  - expected warning here */
enum e7;
enum e7* g7(void);

enum e8 { three };
/** @return e8 Test 8 */
enum e4 *g8(enum e8 *p);
