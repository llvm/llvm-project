#include "status.h"
#include <absl/meta/type_traits.h>
#include <initializer_list>

namespace absl {

template <typename T> struct StatusOr;

namespace internal_statusor {

template <typename T, typename U, typename = void>
struct HasConversionOperatorToStatusOr : std::false_type {};

template <typename T, typename U>
void test(char (*)[sizeof(std::declval<U>().operator absl::StatusOr<T>())]);

template <typename T, typename U>
struct HasConversionOperatorToStatusOr<T, U, decltype(test<T, U>(0))>
    : std::true_type {};

template <typename T, typename U>
using IsConstructibleOrConvertibleFromStatusOr =
    absl::disjunction<std::is_constructible<T, StatusOr<U> &>,
                      std::is_constructible<T, const StatusOr<U> &>,
                      std::is_constructible<T, StatusOr<U> &&>,
                      std::is_constructible<T, const StatusOr<U> &&>,
                      std::is_convertible<StatusOr<U> &, T>,
                      std::is_convertible<const StatusOr<U> &, T>,
                      std::is_convertible<StatusOr<U> &&, T>,
                      std::is_convertible<const StatusOr<U> &&, T>>;

template <typename T, typename U>
using IsConstructibleOrConvertibleOrAssignableFromStatusOr =
    absl::disjunction<IsConstructibleOrConvertibleFromStatusOr<T, U>,
                      std::is_assignable<T &, StatusOr<U> &>,
                      std::is_assignable<T &, const StatusOr<U> &>,
                      std::is_assignable<T &, StatusOr<U> &&>,
                      std::is_assignable<T &, const StatusOr<U> &&>>;

template <typename T, typename U>
struct IsDirectInitializationAmbiguous
    : public absl::conditional_t<
          std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>,
                       U>::value,
          std::false_type,
          IsDirectInitializationAmbiguous<
              T, absl::remove_cv_t<absl::remove_reference_t<U>>>> {};

template <typename T, typename V>
struct IsDirectInitializationAmbiguous<T, absl::StatusOr<V>>
    : public IsConstructibleOrConvertibleFromStatusOr<T, V> {};

template <typename T, typename U>
using IsDirectInitializationValid = absl::disjunction<
    // Short circuits if T is basically U.
    std::is_same<T, absl::remove_cv_t<absl::remove_reference_t<U>>>,
    absl::negation<absl::disjunction<
        std::is_same<absl::StatusOr<T>,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        std::is_same<absl::Status,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        std::is_same<absl::in_place_t,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        IsDirectInitializationAmbiguous<T, U>>>>;

template <typename T, typename U>
struct IsForwardingAssignmentAmbiguous
    : public absl::conditional_t<
          std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>,
                       U>::value,
          std::false_type,
          IsForwardingAssignmentAmbiguous<
              T, absl::remove_cv_t<absl::remove_reference_t<U>>>> {};

template <typename T, typename U>
struct IsForwardingAssignmentAmbiguous<T, absl::StatusOr<U>>
    : public IsConstructibleOrConvertibleOrAssignableFromStatusOr<T, U> {};

template <typename T, typename U>
using IsForwardingAssignmentValid = absl::disjunction<
    // Short circuits if T is basically U.
    std::is_same<T, absl::remove_cv_t<absl::remove_reference_t<U>>>,
    absl::negation<absl::disjunction<
        std::is_same<absl::StatusOr<T>,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        std::is_same<absl::Status,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        std::is_same<absl::in_place_t,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        IsForwardingAssignmentAmbiguous<T, U>>>>;

template <typename T, typename U>
using IsForwardingAssignmentValid = absl::disjunction<
    // Short circuits if T is basically U.
    std::is_same<T, absl::remove_cv_t<absl::remove_reference_t<U>>>,
    absl::negation<absl::disjunction<
        std::is_same<absl::StatusOr<T>,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        std::is_same<absl::Status,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        std::is_same<absl::in_place_t,
                     absl::remove_cv_t<absl::remove_reference_t<U>>>,
        IsForwardingAssignmentAmbiguous<T, U>>>>;

template <typename T> struct OperatorBase {
  const T &value() const &;
  T &value() &;
  const T &&value() const &&;
  T &&value() &&;

  const T &operator*() const &;
  T &operator*() &;
  const T &&operator*() const &&;
  T &&operator*() &&;

  // To test that analyses are okay if there is a use of operator*
  // within this base class.
  const T *operator->() const { return __builtin_addressof(**this); }
  T *operator->() { return __builtin_addressof(**this); }
};

} // namespace internal_statusor

template <typename T>
struct StatusOr : private internal_statusor::OperatorBase<T> {
  explicit StatusOr();

  StatusOr(const StatusOr &) = default;
  StatusOr &operator=(const StatusOr &) = default;

  StatusOr(StatusOr &&) = default;
  StatusOr &operator=(StatusOr &&) = default;

  template <
      typename U,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>,
              std::is_constructible<T, const U &>,
              std::is_convertible<const U &, T>,
              absl::negation<
                  internal_statusor::IsConstructibleOrConvertibleFromStatusOr<
                      T, U>>>::value,
          int> = 0>
  StatusOr(const StatusOr<U> &);

  template <
      typename U,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>,
              std::is_constructible<T, const U &>,
              absl::negation<std::is_convertible<const U &, T>>,
              absl::negation<
                  internal_statusor::IsConstructibleOrConvertibleFromStatusOr<
                      T, U>>>::value,
          int> = 0>
  explicit StatusOr(const StatusOr<U> &);

  template <
      typename U,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>,
              std::is_constructible<T, U &&>, std::is_convertible<U &&, T>,
              absl::negation<
                  internal_statusor::IsConstructibleOrConvertibleFromStatusOr<
                      T, U>>>::value,
          int> = 0>
  StatusOr(StatusOr<U> &&);

  template <
      typename U,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>,
              std::is_constructible<T, U &&>,
              absl::negation<std::is_convertible<U &&, T>>,
              absl::negation<
                  internal_statusor::IsConstructibleOrConvertibleFromStatusOr<
                      T, U>>>::value,
          int> = 0>
  explicit StatusOr(StatusOr<U> &&);

  template <
      typename U,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>,
              std::is_constructible<T, const U &>,
              std::is_assignable<T, const U &>,
              absl::negation<
                  internal_statusor::
                      IsConstructibleOrConvertibleOrAssignableFromStatusOr<
                          T, U>>>::value,
          int> = 0>
  StatusOr &operator=(const StatusOr<U> &);

  template <
      typename U,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_same<T, U>>,
              std::is_constructible<T, U &&>, std::is_assignable<T, U &&>,
              absl::negation<
                  internal_statusor::
                      IsConstructibleOrConvertibleOrAssignableFromStatusOr<
                          T, U>>>::value,
          int> = 0>
  StatusOr &operator=(StatusOr<U> &&);

  template <
      typename U = absl::Status,
      absl::enable_if_t<
          absl::conjunction<
              std::is_convertible<U &&, absl::Status>,
              std::is_constructible<absl::Status, U &&>,
              absl::negation<std::is_same<absl::decay_t<U>, absl::StatusOr<T>>>,
              absl::negation<std::is_same<absl::decay_t<U>, T>>,
              absl::negation<std::is_same<absl::decay_t<U>, absl::in_place_t>>,
              absl::negation<internal_statusor::HasConversionOperatorToStatusOr<
                  T, U &&>>>::value,
          int> = 0>
  StatusOr(U &&);

  template <
      typename U = absl::Status,
      absl::enable_if_t<
          absl::conjunction<
              absl::negation<std::is_convertible<U &&, absl::Status>>,
              std::is_constructible<absl::Status, U &&>,
              absl::negation<std::is_same<absl::decay_t<U>, absl::StatusOr<T>>>,
              absl::negation<std::is_same<absl::decay_t<U>, T>>,
              absl::negation<std::is_same<absl::decay_t<U>, absl::in_place_t>>,
              absl::negation<internal_statusor::HasConversionOperatorToStatusOr<
                  T, U &&>>>::value,
          int> = 0>
  explicit StatusOr(U &&);

  template <
      typename U = absl::Status,
      absl::enable_if_t<
          absl::conjunction<
              std::is_convertible<U &&, absl::Status>,
              std::is_constructible<absl::Status, U &&>,
              absl::negation<std::is_same<absl::decay_t<U>, absl::StatusOr<T>>>,
              absl::negation<std::is_same<absl::decay_t<U>, T>>,
              absl::negation<std::is_same<absl::decay_t<U>, absl::in_place_t>>,
              absl::negation<internal_statusor::HasConversionOperatorToStatusOr<
                  T, U &&>>>::value,
          int> = 0>
  StatusOr &operator=(U &&);

  template <
      typename U = T,
      typename = typename std::enable_if<absl::conjunction<
          std::is_constructible<T, U &&>, std::is_assignable<T &, U &&>,
          absl::disjunction<
              std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>, T>,
              absl::conjunction<
                  absl::negation<std::is_convertible<U &&, absl::Status>>,
                  absl::negation<
                      internal_statusor::HasConversionOperatorToStatusOr<
                          T, U &&>>>>,
          internal_statusor::IsForwardingAssignmentValid<T, U &&>>::value>::
          type>
  StatusOr &operator=(U &&);

  template <typename... Args> explicit StatusOr(absl::in_place_t, Args &&...);

  template <typename U, typename... Args>
  explicit StatusOr(absl::in_place_t, std::initializer_list<U>, Args &&...);

  template <
      typename U = T,
      absl::enable_if_t<
          absl::conjunction<
              internal_statusor::IsDirectInitializationValid<T, U &&>,
              std::is_constructible<T, U &&>, std::is_convertible<U &&, T>,
              absl::disjunction<
                  std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>,
                               T>,
                  absl::conjunction<
                      absl::negation<std::is_convertible<U &&, absl::Status>>,
                      absl::negation<
                          internal_statusor::HasConversionOperatorToStatusOr<
                              T, U &&>>>>>::value,
          int> = 0>
  StatusOr(U &&);

  template <
      typename U = T,
      absl::enable_if_t<
          absl::conjunction<
              internal_statusor::IsDirectInitializationValid<T, U &&>,
              absl::disjunction<
                  std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>,
                               T>,
                  absl::conjunction<
                      absl::negation<std::is_constructible<absl::Status, U &&>>,
                      absl::negation<
                          internal_statusor::HasConversionOperatorToStatusOr<
                              T, U &&>>>>,
              std::is_constructible<T, U &&>,
              absl::negation<std::is_convertible<U &&, T>>>::value,
          int> = 0>
  explicit StatusOr(U &&);

  bool ok() const;

  const Status &status() const & { return status_; }
  Status status() &&;

  using StatusOr::OperatorBase::value;

  const T &ValueOrDie() const &;
  T &ValueOrDie() &;
  const T &&ValueOrDie() const &&;
  T &&ValueOrDie() &&;

  using StatusOr::OperatorBase::operator*;
  using StatusOr::OperatorBase::operator->;

  template <typename U> T value_or(U &&default_value) const &;
  template <typename U> T value_or(U &&default_value) &&;

  template <typename... Args> T &emplace(Args &&...args);

  template <
      typename U, typename... Args,
      absl::enable_if_t<std::is_constructible<T, std::initializer_list<U> &,
                                              Args &&...>::value,
                        int> = 0>
  T &emplace(std::initializer_list<U> ilist, Args &&...args);

private:
  absl::Status status_;
};

template <typename T>
bool operator==(const StatusOr<T> &lhs, const StatusOr<T> &rhs);

template <typename T>
bool operator!=(const StatusOr<T> &lhs, const StatusOr<T> &rhs);

} // namespace absl
