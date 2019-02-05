// Mimic the implementation of absl::Duration
namespace absl {

using int64_t = long long int;

class Duration {
public:
  Duration &operator*=(int64_t r);
  Duration &operator*=(float r);
  Duration &operator*=(double r);
  template <typename T> Duration &operator*=(T r);

  Duration &operator/=(int64_t r);
  Duration &operator/=(float r);
  Duration &operator/=(double r);
  template <typename T> Duration &operator/=(T r);

  Duration &operator+(Duration d);
};

template <typename T> Duration operator*(Duration lhs, T rhs);
template <typename T> Duration operator*(T lhs, Duration rhs);
template <typename T> Duration operator/(Duration lhs, T rhs);

class Time{};

constexpr Duration Nanoseconds(long long);
constexpr Duration Microseconds(long long);
constexpr Duration Milliseconds(long long);
constexpr Duration Seconds(long long);
constexpr Duration Minutes(long long);
constexpr Duration Hours(long long);

template <typename T> struct EnableIfFloatImpl {};
template <> struct EnableIfFloatImpl<float> { typedef int Type; };
template <> struct EnableIfFloatImpl<double> { typedef int Type; };
template <> struct EnableIfFloatImpl<long double> { typedef int Type; };
template <typename T> using EnableIfFloat = typename EnableIfFloatImpl<T>::Type;

template <typename T, EnableIfFloat<T> = 0> Duration Nanoseconds(T n);
template <typename T, EnableIfFloat<T> = 0> Duration Microseconds(T n);
template <typename T, EnableIfFloat<T> = 0> Duration Milliseconds(T n);
template <typename T, EnableIfFloat<T> = 0> Duration Seconds(T n);
template <typename T, EnableIfFloat<T> = 0> Duration Minutes(T n);
template <typename T, EnableIfFloat<T> = 0> Duration Hours(T n);

double ToDoubleHours(Duration d);
double ToDoubleMinutes(Duration d);
double ToDoubleSeconds(Duration d);
double ToDoubleMilliseconds(Duration d);
double ToDoubleMicroseconds(Duration d);
double ToDoubleNanoseconds(Duration d);
int64_t ToInt64Hours(Duration d);
int64_t ToInt64Minutes(Duration d);
int64_t ToInt64Seconds(Duration d);
int64_t ToInt64Milliseconds(Duration d);
int64_t ToInt64Microseconds(Duration d);
int64_t ToInt64Nanoseconds(Duration d);

int64_t ToUnixHours(Time t);
int64_t ToUnixMinutes(Time t);
int64_t ToUnixSeconds(Time t);
int64_t ToUnixMillis(Time t);
int64_t ToUnixMicros(Time t);
int64_t ToUnixNanos(Time t);
Time FromUnixHours(int64_t);
Time FromUnixMinutes(int64_t);
Time FromUnixSeconds(int64_t);
Time FromUnixMillis(int64_t);
Time FromUnixMicros(int64_t);
Time FromUnixNanos(int64_t);

// Relational Operators
constexpr bool operator<(Duration lhs, Duration rhs);
constexpr bool operator>(Duration lhs, Duration rhs);
constexpr bool operator>=(Duration lhs, Duration rhs);
constexpr bool operator<=(Duration lhs, Duration rhs);
constexpr bool operator==(Duration lhs, Duration rhs);
constexpr bool operator!=(Duration lhs, Duration rhs);

// Additive Operators
inline Time operator+(Time lhs, Duration rhs);
inline Time operator+(Duration lhs, Time rhs);
inline Time operator-(Time lhs, Duration rhs);
inline Duration operator-(Time lhs, Time rhs);

}  // namespace absl
