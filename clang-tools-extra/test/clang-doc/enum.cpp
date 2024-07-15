// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --doxygen --output=%t --executor=standalone %s
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/GlobalNamespace/index.html -check-prefix=HTML-INDEX
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.html -check-prefix=HTML-ANIMAL
// RUN: FileCheck %s < %t/Vehicles/index.html -check-prefix=HTML-VEHICLES
// RUN: FileCheck %s < %t/GlobalNamespace/index.md -check-prefix=MD-INDEX
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.md -check-prefix=MD-ANIMAL
// RUN: FileCheck %s < %t/Vehicles/index.md -check-prefix=MD-VEHICLES



// MD-INDEX: # Global Namespace
// MD-INDEX: ## Enums
// MD-INDEX: | enum Color |
// MD-INDEX: --
// MD-INDEX: | Red |
// MD-INDEX: | Green |
// MD-INDEX: | Blue |
// MD-INDEX: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE+13]]*
// MD-INDEX: **brief** For specifying RGB colors

// HTML-INDEX: <h1>Global Namespace</h1>
// HTML-INDEX: <h2 id="Enums">Enums</h2>
// HTML-INDEX: <h3 id="{{([0-9A-F]{40})}}">enum Color</h3>
// HTML-INDEX: <li>Red</li>
// HTML-INDEX: <li>Green</li>
// HTML-INDEX: <li>Blue</li>
// HTML-INDEX: <p>Defined at line [[@LINE+4]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
/**
 * @brief For specifying RGB colors
 */
enum Color {
  Red, ///< Red
  Green, ///< Green
  Blue ///< Blue
};


// MD-INDEX: | enum class Shapes |
// MD-INDEX: --
// MD-INDEX: | Circle |
// MD-INDEX: | Rectangle |
// MD-INDEX: | Triangle |
// MD-INDEX: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE+11]]*
// MD-INDEX: **brief** Shape Types

// HTML-INDEX: <h3 id="{{([0-9A-F]{40})}}">enum class Shapes</h3>
// HTML-INDEX: <li>Circle</li>
// HTML-INDEX: <li>Rectangle</li>
// HTML-INDEX: <li>Triangle</li>
// HTML-INDEX: <p>Defined at line [[@LINE+4]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
/**
 * @brief Shape Types
 */
enum class Shapes {
  /// Circle
  Circle,
  /// Rectangle
  Rectangle,
  /// Triangle
  Triangle
};


// MD-ANIMAL: # class Animals
// MD-ANIMAL: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE+18]]*
// MD-ANIMAL: ## Enums
// MD-ANIMAL: | enum AnimalType |
// MD-ANIMAL: --
// MD-ANIMAL: | Dog |
// MD-ANIMAL: | Cat |
// MD-ANIMAL: | Iguana |
// MD-ANIMAL: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE+16]]*
// MD-ANIMAL: **brief** specify what animal the class is

// HTML-ANIMAL: <h1>class Animals</h1>
// HTML-ANIMAL: <p>Defined at line [[@LINE+7]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
// HTML-ANIMAL: <h2 id="Enums">Enums</h2>
// HTML-ANIMAL: <h3 id="{{([0-9A-F]{40})}}">enum AnimalType</h3>
// HTML-ANIMAL: <li>Dog</li>
// HTML-ANIMAL: <li>Cat</li>
// HTML-ANIMAL: <li>Iguana</li>
// HTML-ANIMAL: <p>Defined at line [[@LINE+6]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
class Animals {
public:
      /**
       * @brief specify what animal the class is
       */
      enum AnimalType {
          Dog, /// Man's best friend
          Cat, /// Man's other best friend
          Iguana /// A lizard
      };
};



// MD-VEHICLES: # namespace Vehicles
// MD-VEHICLES: ## Enums
// MD-VEHICLES: | enum Car |
// MD-VEHICLES: --
// MD-VEHICLES: | Sedan |
// MD-VEHICLES: | SUV |
// MD-VEHICLES: | Pickup |
// MD-VEHICLES: | Hatchback |
// MD-VEHICLES: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE+15]]*
// MD-VEHICLES: **brief** specify type of car

// HTML-VEHICLES: <h1>namespace Vehicles</h1>
// HTML-VEHICLES: <h2 id="Enums">Enums</h2>
// HTML-VEHICLES: <h3 id="{{([0-9A-F]{40})}}">enum Car</h3>
// HTML-VEHICLES: <li>Sedan</li>
// HTML-VEHICLES: <li>SUV</li>
// HTML-VEHICLES: <li>Pickup</li>
// HTML-VEHICLES: <li>Hatchback</li>
// HTML-VEHICLES: <p>Defined at line [[@LINE+5]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
namespace Vehicles {
    /**
     * @brief specify type of car
     */
    enum Car {
       Sedan, /// Sedan
       SUV, /// SUV
       Pickup, /// Pickup
       Hatchback /// Hatchback
    };
}