// RUN: %clang_cc1 %s -E -fms-compatibility

bool f() {
  // Check that operators still work before redefining them.
#if compl 0 bitand 1
  return true and false;
#endif
}

// All c++ keywords should be #define-able in ms mode.
// (operators like "and" aren't normally, the rest always is.)
#define and
#define and_eq
#define alignas
#define alignof
#define asm
#define auto
#define bitand
#define bitor
#define bool
#define break
#define case
#define catch
#define char
#define char16_t
#define char32_t
#define class
#define compl
#define const
#define constexpr
#define const_cast
#define continue
#define decltype
#define default
#define delete
#define double
#define dynamic_cast
#define else
#define enum
#define explicit
#define export
#define extern
#define false
#define float
#define for
#define friend
#define goto
#define if
#define inline
#define int
#define long
#define mutable
#define namespace
#define new
#define noexcept
#define not
#define not_eq
#define nullptr
#define operator
#define or
#define or_eq
#define private
#define protected
#define public
#define register
#define reinterpret_cast
#define return
#define short
#define signed
#define sizeof
#define static
#define static_assert
#define static_cast
#define struct
#define switch
#define template
#define this
#define thread_local
#define throw
#define true
#define try
#define typedef
#define typeid
#define typename
#define union
#define unsigned
#define using
#define virtual
#define void
#define volatile
#define wchar_t
#define while
#define xor
#define xor_eq

// Check this is all properly defined away.
and
and_eq
alignas
alignof
asm
auto
bitand
bitor
bool
break
case
catch
char
char16_t
char32_t
class
compl
const
constexpr
const_cast
continue
decltype
default
delete
double
dynamic_cast
else
enum
explicit
export
extern
false
float
for
friend
goto
if
inline
int
long
mutable
namespace
new
noexcept
not
not_eq
nullptr
operator
or
or_eq
private
protected
public
register
reinterpret_cast
return
short
signed
sizeof
static
static_assert
static_cast
struct
switch
template
this
thread_local
throw
true
try
typedef
typeid
typename
union
unsigned
using
virtual
void
volatile
wchar_t
while
xor
xor_eq
