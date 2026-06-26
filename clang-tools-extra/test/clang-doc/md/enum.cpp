// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %S/../Inputs/enum.cpp
// RUN: clang-doc --format=md_mustache --pretty-json --doxygen --output=%t --executor=standalone %S/../Inputs/enum.cpp
// RUN: FileCheck %s < %t/GlobalNamespace/index.md --check-prefix=MD-INDEX
// RUN: FileCheck %s < %t/GlobalNamespace/Animals.md --check-prefix=MD-ANIMAL
// RUN: FileCheck %s < %t/GlobalNamespace/FilePermissions.md --check-prefix=MD-PERM
// RUN: FileCheck %s < %t/Vehicles/index.md --check-prefix=MD-VEHICLES
// RUN: FileCheck %s < %t/md/GlobalNamespace/index.md --check-prefix=MD-MUSTACHE-INDEX
// RUN: FileCheck %s < %t/md/GlobalNamespace/_ZTV7Animals.md --check-prefix=MD-MUSTACHE-ANIMAL
// RUN: FileCheck %s < %t/md/Vehicles/index.md --check-prefix=MD-MUSTACHE-VEHICLES

// MD-INDEX: ## Enums
// MD-INDEX: | enum Color |
// MD-INDEX: | Name | Value | Comments |
// MD-INDEX: |---|---|---|
// MD-INDEX: | Red | 0 | Comment 1 |
// MD-INDEX: | Green | 1 | Comment 2 |
// MD-INDEX: | Blue | 2 | Comment 3 |
// MD-INDEX: **brief** For specifying RGB colors

// MD-MUSTACHE-INDEX: ## Enums
// MD-MUSTACHE-INDEX: | enum Color |
// MD-MUSTACHE-INDEX: --
// MD-MUSTACHE-INDEX: | Red |
// MD-MUSTACHE-INDEX: | Green |
// MD-MUSTACHE-INDEX: | Blue |
// MD-MUSTACHE-INDEX: **brief** For specifying RGB colors

// MD-INDEX: | enum class Shapes |
// MD-INDEX: | Name | Value | Comments |
// MD-INDEX: |---|---|---|
// MD-INDEX: | Circle | 0 | Comment 1 |
// MD-INDEX: | Rectangle | 1 | Comment 2 |
// MD-INDEX: | Triangle | 2 | Comment 3 |
// MD-INDEX: **brief** Shape Types

// MD-INDEX: | enum Size : uint8_t |
// MD-INDEX: | Name | Value | Comments |
// MD-INDEX: |---|---|---|
// MD-INDEX: | Small | 0 | A pearl.<br>Pearls are quite small.<br><br>Pearls are used in jewelry. |
// MD-INDEX: | Medium | 1 | A tennis ball. |
// MD-INDEX: | Large | 2 | A football. |
// MD-INDEX: **brief** Specify the size

// MD-INDEX: | enum (unnamed) : long long |
// MD-INDEX: | Name | Value | Comments |
// MD-INDEX: |---|---|---|
// MD-INDEX: | BigVal | 999999999999 | A very large value |
// MD-INDEX: **brief** Very long number

// MD-INDEX: | enum ColorUserSpecified |
// MD-INDEX: | Name | Value |
// MD-INDEX: |---|---|
// MD-INDEX: | RedUserSpecified | 65 |
// MD-INDEX: | GreenUserSpecified | 2 |
// MD-INDEX: | BlueUserSpecified | 67 |

// MD-MUSTACHE-INDEX: | enum ColorUserSpecified |
// MD-MUSTACHE-INDEX: --
// MD-MUSTACHE-INDEX: | RedUserSpecified |
// MD-MUSTACHE-INDEX: | GreenUserSpecified |
// MD-MUSTACHE-INDEX: | BlueUserSpecified |

// MD-PERM: | enum (unnamed) |
// MD-PERM: | Name | Value | Comments |
// MD-PERM: |---|---|---|
// MD-PERM: | Read | 1 | Permission to READ r |
// MD-PERM: | Write | 2 | Permission to WRITE w |
// MD-PERM: | Execute | 4 | Permission to EXECUTE x |
// MD-PERM: **brief** File permission flags

// MD-ANIMAL: # class Animals
// MD-ANIMAL: ## Enums
// MD-ANIMAL: | enum AnimalType |
// MD-ANIMAL: | Name | Value | Comments |
// MD-ANIMAL: |---|---|---|
// MD-ANIMAL: | Dog | 0 | Man's best friend |
// MD-ANIMAL: | Cat | 1 | Man's other best friend |
// MD-ANIMAL: | Iguana | 2 | A lizard |
// MD-ANIMAL: **brief** specify what animal the class is

// MD-MUSTACHE-ANIMAL: # class Animals
// MD-MUSTACHE-ANIMAL: ## Enums
// MD-MUSTACHE-ANIMAL: | enum AnimalType |
// MD-MUSTACHE-ANIMAL: --
// MD-MUSTACHE-ANIMAL: | Dog |
// MD-MUSTACHE-ANIMAL: | Cat |
// MD-MUSTACHE-ANIMAL: | Iguana |
// MD-MUSTACHE-ANIMAL: **brief** specify what animal the class is

// MD-VEHICLES: # namespace Vehicles
// MD-VEHICLES: ## Enums
// MD-VEHICLES: | enum Car |
// MD-VEHICLES: | Name | Value | Comments |
// MD-VEHICLES: |---|---|---|
// MD-VEHICLES: | Sedan | 0 | Comment 1 |
// MD-VEHICLES: | SUV | 1 | Comment 2 |
// MD-VEHICLES: | Pickup | 2 | -- |
// MD-VEHICLES: | Hatchback | 3 | Comment 4 |
// MD-VEHICLES: **brief** specify type of car

// MD-MUSTACHE-VEHICLES: # namespace Vehicles
// MD-MUSTACHE-VEHICLES: ## Enums
// MD-MUSTACHE-VEHICLES: | enum Car |
// MD-MUSTACHE-VEHICLES: --
// MD-MUSTACHE-VEHICLES: | Sedan |
// MD-MUSTACHE-VEHICLES: | SUV |
// MD-MUSTACHE-VEHICLES: | Pickup |
// MD-MUSTACHE-VEHICLES: | Hatchback |
// MD-MUSTACHE-VEHICLES: **brief** specify type of car
