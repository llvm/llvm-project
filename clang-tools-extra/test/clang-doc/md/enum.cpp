// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=md --pretty-json --doxygen --output=%t --executor=standalone %S/../Inputs/enum.cpp
// RUN: FileCheck %s < %t/md/GlobalNamespace/index.md --check-prefix=MD-INDEX
// RUN: FileCheck %s < %t/md/GlobalNamespace/_ZTV7Animals.md --check-prefix=MD-ANIMAL
// RUN: FileCheck %s < %t/md/Vehicles/index.md --check-prefix=MD-VEHICLES

// MD-INDEX: ## Enums
// MD-INDEX: | enum Color |
// MD-INDEX: --
// MD-INDEX: | Red |
// MD-INDEX: | Green |
// MD-INDEX: | Blue |
// MD-INDEX: **brief** For specifying RGB colors

// MD-INDEX: | enum ColorUserSpecified |
// MD-INDEX: --
// MD-INDEX: | RedUserSpecified |
// MD-INDEX: | GreenUserSpecified |
// MD-INDEX: | BlueUserSpecified |

// MD-ANIMAL: # class Animals
// MD-ANIMAL: ## Enums
// MD-ANIMAL: | enum AnimalType |
// MD-ANIMAL: --
// MD-ANIMAL: | Dog |
// MD-ANIMAL: | Cat |
// MD-ANIMAL: | Iguana |
// MD-ANIMAL: **brief** specify what animal the class is

// MD-VEHICLES: # namespace Vehicles
// MD-VEHICLES: ## Enums
// MD-VEHICLES: | enum Car |
// MD-VEHICLES: --
// MD-VEHICLES: | Sedan |
// MD-VEHICLES: | SUV |
// MD-VEHICLES: | Pickup |
// MD-VEHICLES: | Hatchback |
// MD-VEHICLES: **brief** specify type of car
