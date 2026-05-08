/**
 * @brief For specifying RGB colors
 */
enum Color {
  Red,   ///< Comment 1
  Green, ///< Comment 2
  Blue   ///< Comment 3
};

/**
 * @brief Shape Types
 */
enum class Shapes {
  /// Comment 1
  Circle,
  /// Comment 2
  Rectangle,
  /// Comment 3
  Triangle
};

typedef unsigned char uint8_t;
/**
 * @brief Specify the size
 */
enum Size : uint8_t {
  /// A pearl.
  /// Pearls are quite small.
  ///
  /// Pearls are used in jewelry.
  Small,

  /// @brief A tennis ball.
  Medium,

  /// A football.
  Large
};

/**
 * @brief Very long number
 */
enum : long long {
  BigVal = 999999999999   ///< A very large value
};

enum ColorUserSpecified {
  RedUserSpecified = 'A',
  GreenUserSpecified = 2,
  BlueUserSpecified = 'C'
};

class FilePermissions {
public:
  /**
   * @brief File permission flags
   */
  enum {
    Read    = 1,     ///< Permission to READ r
    Write   = 2,     ///< Permission to WRITE w
    Execute = 4      ///< Permission to EXECUTE x
  };
};

// COM: FIXME: Add enums declared inside of classes to class template
class Animals {
public:
  /**
   * @brief specify what animal the class is
   */
  enum AnimalType {
    Dog,   ///< Man's best friend
    Cat,   ///< Man's other best friend
    Iguana ///< A lizard
  };
};

namespace Vehicles {
/**
 * @brief specify type of car
 */
enum Car {
  Sedan,    ///< Comment 1
  SUV,      ///< Comment 2
  Pickup,
  Hatchback ///< Comment 4
};
} // namespace Vehicles
