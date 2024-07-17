// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --doxygen --output=%t --executor=standalone %s
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/GlobalNamespace/index.html --check-prefix=HTML-INDEX-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/index.html --check-prefix=HTML-INDEX
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.html --check-prefix=HTML-ANIMAL-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.html --check-prefix=HTML-ANIMAL
// RUN: FileCheck %s < %t/Vehicles/index.html --check-prefix=HTML-VEHICLES-LINE
// RUN: FileCheck %s < %t/Vehicles/index.html --check-prefix=HTML-VEHICLES
// RUN: FileCheck %s < %t/GlobalNamespace/index.md --check-prefix=MD-INDEX-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/index.md --check-prefix=MD-INDEX
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.md --check-prefix=MD-ANIMAL-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.md --check-prefix=MD-ANIMAL
// RUN: FileCheck %s < %t/Vehicles/index.md --check-prefix=MD-VEHICLES-LINE
// RUN: FileCheck %s < %t/Vehicles/index.md --check-prefix=MD-VEHICLES


/**
 * @brief For specifying RGB colors
 */
enum Color {
// MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
// HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  Red, ///< Red
  Green, ///< Green
  Blue ///< Blue
};

// MD-INDEX: ## Enums
// MD-INDEX: | enum Color |
// MD-INDEX: --
// MD-INDEX: | Red |
// MD-INDEX: | Green |
// MD-INDEX: | Blue |
// MD-INDEX: **brief** For specifying RGB colors

// HTML-INDEX: <h2 id="Enums">Enums</h2>
// HTML-INDEX: <h3 id="{{([0-9A-F]{40})}}">enum Color</h3>
// HTML-INDEX: <li>Red</li>
// HTML-INDEX: <li>Green</li>
// HTML-INDEX: <li>Blue</li>

/**
 * @brief Shape Types
 */
enum class Shapes {
// MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
// HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
  /// Circle
  Circle,
  /// Rectangle
  Rectangle,
  /// Triangle
  Triangle
};
// MD-INDEX: | enum class Shapes |
// MD-INDEX: --
// MD-INDEX: | Circle |
// MD-INDEX: | Rectangle |
// MD-INDEX: | Triangle |
// MD-INDEX: **brief** Shape Types

// HTML-INDEX: <h3 id="{{([0-9A-F]{40})}}">enum class Shapes</h3>
// HTML-INDEX: <li>Circle</li>
// HTML-INDEX: <li>Rectangle</li>
// HTML-INDEX: <li>Triangle</li>


class Animals {
// MD-ANIMAL-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
// HTML-ANIMAL-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
public:
      /**
       * @brief specify what animal the class is
       */
      enum AnimalType {
// MD-ANIMAL-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
// HTML-ANIMAL-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
          Dog, /// Man's best friend
          Cat, /// Man's other best friend
          Iguana /// A lizard
      };
};

// HTML-ANIMAL: <h1>class Animals</h1>
// HTML-ANIMAL: <h2 id="Enums">Enums</h2>
// HTML-ANIMAL: <h3 id="{{([0-9A-F]{40})}}">enum AnimalType</h3>
// HTML-ANIMAL: <li>Dog</li>
// HTML-ANIMAL: <li>Cat</li>
// HTML-ANIMAL: <li>Iguana</li>

// MD-ANIMAL: # class Animals
// MD-ANIMAL: ## Enums
// MD-ANIMAL: | enum AnimalType |
// MD-ANIMAL: --
// MD-ANIMAL: | Dog |
// MD-ANIMAL: | Cat |
// MD-ANIMAL: | Iguana |
// MD-ANIMAL: **brief** specify what animal the class is


namespace Vehicles {
    /**
     * @brief specify type of car
     */
    enum Car {
// MD-VEHICLES-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
// HTML-VEHICLES-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>
       Sedan, /// Sedan
       SUV, /// SUV
       Pickup, /// Pickup
       Hatchback /// Hatchback
    };
}

// MD-VEHICLES: # namespace Vehicles
// MD-VEHICLES: ## Enums
// MD-VEHICLES: | enum Car |
// MD-VEHICLES: --
// MD-VEHICLES: | Sedan |
// MD-VEHICLES: | SUV |
// MD-VEHICLES: | Pickup |
// MD-VEHICLES: | Hatchback |
// MD-VEHICLES: **brief** specify type of car

// HTML-VEHICLES: <h1>namespace Vehicles</h1>
// HTML-VEHICLES: <h2 id="Enums">Enums</h2>
// HTML-VEHICLES: <h3 id="{{([0-9A-F]{40})}}">enum Car</h3>
// HTML-VEHICLES: <li>Sedan</li>
// HTML-VEHICLES: <li>SUV</li>
// HTML-VEHICLES: <li>Pickup</li>
// HTML-VEHICLES: <li>Hatchback</li>