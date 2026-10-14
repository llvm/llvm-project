namespace absl {
struct SourceLocation {
  static constexpr SourceLocation current();
  static constexpr SourceLocation
  DoNotInvokeDirectlyNoSeriouslyDont(int line, const char *file_name);
};
} // namespace absl
namespace absl {
enum class StatusCode : int {
  kOk,
  kCancelled,
  kUnknown,
  kInvalidArgument,
  kDeadlineExceeded,
  kNotFound,
  kAlreadyExists,
  kPermissionDenied,
  kResourceExhausted,
  kFailedPrecondition,
  kAborted,
  kOutOfRange,
  kUnimplemented,
  kInternal,
  kUnavailable,
  kDataLoss,
  kUnauthenticated,
};
} // namespace absl

namespace absl {
enum class StatusToStringMode : int {
  kWithNoExtraData = 0,
  kWithPayload = 1 << 0,
  kWithSourceLocation = 1 << 1,
  kWithEverything = ~kWithNoExtraData,
  kDefault = kWithPayload,
};
class Status {
public:
  Status();
  Status(const Status &base_status, absl::SourceLocation loc);
  Status(Status &&base_status, absl::SourceLocation loc);
  ~Status() {}

  Status(const Status &);
  Status &operator=(const Status &x);

  Status(Status &&) noexcept;
  Status &operator=(Status &&);

  friend bool operator==(const Status &, const Status &);
  friend bool operator!=(const Status &, const Status &);

  bool ok() const { return true; }
  void CheckSuccess() const;
  void IgnoreError() const;
  int error_code() const;
  absl::Status ToCanonical() const;
  void Update(const Status &new_status);
  void Update(Status &&new_status);
};

bool operator==(const Status &lhs, const Status &rhs);
bool operator!=(const Status &lhs, const Status &rhs);

Status OkStatus();
Status InvalidArgumentError(const char *);

} // namespace absl
