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
  Red, ///< Comment 1
  Green, ///< Comment 2
  Blue ///< Comment 3
};

// MD-INDEX: ## Enums
// MD-INDEX: | enum Color |
// MD-INDEX: --
// MD-INDEX: | Red |
// MD-INDEX: | Green |
// MD-INDEX: | Blue |
// MD-INDEX: **brief** For specifying RGB colors

// HTML-INDEX: <th colspan="3">enum Color</th>
// HTML-INDEX: <td>Red</td>
// HTML-INDEX: <td>0</td>
// HTML-INDEX: <p> Comment 1</p>
// HTML-INDEX: <td>Green</td>
// HTML-INDEX: <td>1</td>
// HTML-INDEX: <p> Comment 2</p>
// HTML-INDEX: <td>Blue</td>
// HTML-INDEX: <td>2</td>
// HTML-INDEX: <p> Comment 3</p>

/**
 * @brief Shape Types
 */
enum class Shapes {
// MD-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp#[[@LINE-1]]*
// HTML-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}enum.cpp</p>

  /// Comment 1
  Circle,
  /// Comment 2
  Rectangle,
  /// Comment 3
  Triangle
};
// MD-INDEX: | enum class Shapes |
// MD-INDEX: --
// MD-INDEX: | Circle |
// MD-INDEX: | Rectangle |
// MD-INDEX: | Triangle |
// MD-INDEX: **brief** Shape Types

// HTML-INDEX: <th colspan="3">enum class Shapes</th>
// HTML-INDEX: <td>Circle</td>
// HTML-INDEX: <td>0</td>
// HTML-INDEX: <p> Comment 1</p>
// HTML-INDEX: <td>Rectangle</td>
// HTML-INDEX: <td>1</td>
// HTML-INDEX: <p> Comment 2</p>
// HTML-INDEX: <td>Triangle</td>
// HTML-INDEX: <td>2</td>
// HTML-INDEX: <p> Comment 3</p>



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
          Dog, ///< Man's best friend
          Cat, ///< Man's other best friend
          Iguana ///< A lizard
      };
};

// HTML-ANIMAL: <h1>class Animals</h1>
// HTML-ANIMAL: <h2 id="Enums">Enums</h2>
// HTML-ANIMAL: <th colspan="3">enum AnimalType</th>
// HTML-ANIMAL: <td>Dog</td>
// HTML-ANIMAL: <td>0</td>
// HTML-ANIMAL: <p> Man&apos;s best friend</p>
// HTML-ANIMAL: <td>Cat</td>
// HTML-ANIMAL: <td>1</td>
// HTML-ANIMAL: <p> Man&apos;s other best friend</p>
// HTML-ANIMAL: <td>Iguana</td>
// HTML-ANIMAL: <td>2</td>
// HTML-ANIMAL: <p> A lizard</p>


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

       Sedan, ///< Comment 1
       SUV, ///< Comment 2
       Pickup, ///< Comment 3
       Hatchback ///< Comment 4
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
// HTML-VEHICLES: <th colspan="3">enum Car</th>
// HTML-VEHICLES: <td>Sedan</td>
// HTML-VEHICLES: <td>0</td>
// HTML-VEHICLES: <p> Comment 1</p>
// HTML-VEHICLES: <td>SUV</td>
// HTML-VEHICLES: <td>1</td>
// HTML-VEHICLES: <p> Comment 2</p>
// HTML-VEHICLES: <td>Pickup</td>
// HTML-VEHICLES: <td>2</td>
// HTML-VEHICLES: <p> Comment 3</p>
// HTML-VEHICLES: <td>Hatchback</td>
// HTML-VEHICLES: <td>3</td>
// HTML-VEHICLES: <p> Comment 4</p>


enum ColorUserSpecified {
  RedUserSpecified = 'A',
  GreenUserSpecified = 2,
  BlueUserSpecified = 'C'
};

// MD-INDEX: | enum ColorUserSpecified |
// MD-INDEX: --
// MD-INDEX: | RedUserSpecified |
// MD-INDEX: | GreenUserSpecified |
// MD-INDEX: | BlueUserSpecified |

// HTML-INDEX: <th colspan="2">enum ColorUserSpecified</th>
// HTML-INDEX: <td>RedUserSpecified</td>
// HTML-INDEX: <td>&apos;A&apos;</td>
// HTML-INDEX: <td>GreenUserSpecified</td>
// HTML-INDEX: <td>2</td>
// HTML-INDEX: <td>BlueUserSpecified</td>
// HTML-INDEX: <td>&apos;C&apos;</td>
