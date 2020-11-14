// RUN: %check_clang_tidy %s bugprone-redundant-branch-condition %t

extern unsigned peopleInTheBuilding;
extern unsigned fireFighters;

bool isBurning();
bool isReallyBurning();
bool isCollapsing();
void tryToExtinguish(bool&);
void tryPutFireOut();
bool callTheFD();
void scream();

bool someOtherCondition();

//===--- Basic Positives --------------------------------------------------===//

void positive_direct() {
  bool onFire = isBurning();
  if (onFire) {
    if (onFire) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: {{^\ *$}}
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
  }
}

void positive_indirect() {
  bool onFire = isBurning();
  if (onFire) {
    if (someOtherCondition()) {
      if (onFire)
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: {{^\ *$}}
        scream();
    }
  }
}

void positive_direct_inner_and_lhs() {
  bool onFire = isBurning();
  if (onFire) {
    if (onFire && peopleInTheBuilding > 0) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: if ( peopleInTheBuilding > 0) {
      scream();
    }
  }
}

void positive_indirect_inner_and_lhs() {
  bool onFire = isBurning();
  if (onFire) {
    if (someOtherCondition()) {
      if (onFire && peopleInTheBuilding > 0) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: if ( peopleInTheBuilding > 0) {
        scream();
      }
    }
  }
}

void positive_direct_inner_and_rhs() {
  bool onFire = isBurning();
  if (onFire) {
    if (peopleInTheBuilding > 0 && onFire) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: if (peopleInTheBuilding > 0 ) {
      scream();
    }
  }
}

void positive_indirect_inner_and_rhs() {
  bool onFire = isBurning();
  if (onFire) {
    if (someOtherCondition()) {
      if (peopleInTheBuilding > 0 && onFire) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: if (peopleInTheBuilding > 0 ) {
        scream();
      }
    }
  }
}

void positive_direct_inner_or_lhs() {
  bool onFire = isBurning();
  if (onFire) {
    if (onFire || isCollapsing()) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: {{^\ *$}}
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
  }
}

void positive_indirect_inner_or_lhs() {
  bool onFire = isBurning();
  if (onFire) {
    if (someOtherCondition()) {
      if (onFire || isCollapsing()) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: {{^\ *$}}
        scream();
      }
      // CHECK-FIXES: {{^\ *$}}
    }
  }
}

void positive_direct_inner_or_rhs() {
  bool onFire = isBurning();
  if (onFire) {
    if (isCollapsing() || onFire) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: {{^\ *$}}
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
  }
}

void positive_indirect_inner_or_rhs() {
  bool onFire = isBurning();
  if (onFire) {
    if (someOtherCondition()) {
      if (isCollapsing() || onFire) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: {{^\ *$}}
        scream();
      }
      // CHECK-FIXES: {{^\ *$}}
    }
  }
}

void positive_direct_outer_and_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (onFire) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
    // CHECK-FIXES: {{^\ *$}}
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
  }
}

void positive_indirect_outer_and_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (someOtherCondition()) {
      if (onFire) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: {{^\ *$}}
        scream();
      }
      // CHECK-FIXES: {{^\ *$}}
    }
  }
}

void positive_direct_outer_and_lhs_inner_and_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (onFire && peopleInTheBuilding > 0) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: if ( peopleInTheBuilding > 0) {
      scream();
    }
  }
}

void positive_indirect_outer_and_lhs_inner_and_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (someOtherCondition()) {
      if (onFire && peopleInTheBuilding > 0) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: if ( peopleInTheBuilding > 0) {
        scream();
      }
    }
  }
}

void positive_direct_outer_and_lhs_inner_and_rhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (peopleInTheBuilding > 0 && onFire) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: if (peopleInTheBuilding > 0 ) {
      scream();
    }
  }
}

void positive_indirect_outer_and_lhs_inner_and_rhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (someOtherCondition()) {
      if (peopleInTheBuilding > 0 && onFire) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: if (peopleInTheBuilding > 0 ) {
        scream();
      }
    }
  }
}

void positive_direct_outer_and_lhs_inner_or_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (onFire || isCollapsing()) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: {{^\ *$}}
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
  }
}

void positive_indirect_outer_and_lhs_inner_or_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (someOtherCondition()) {
      if (onFire || isCollapsing()) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: {{^\ *$}}
        scream();
      }
    // CHECK-FIXES: {{^\ *$}}
    }
  }
}

void positive_direct_outer_and_lhs_inner_or_rhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (isCollapsing() || onFire) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: {{^\ *$}}
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
  }
}

void positive_indirect_outer_and_lhs_inner_or_rhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (someOtherCondition()) {
      if (isCollapsing() || onFire) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: {{^\ *$}}
        scream();
      }
      // CHECK-FIXES: {{^\ *$}}
    }
  }
}

void positive_direct_outer_and_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (onFire) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: {{^\ *$}}
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
  }
}

void positive_indirect_outer_and_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (someOtherCondition()) {
      if (onFire) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: {{^\ *$}}
        scream();
      }
      // CHECK-FIXES: {{^\ *$}}
    }
  }
}

void positive_direct_outer_and_rhs_inner_and_lhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (onFire && peopleInTheBuilding > 0) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: if ( peopleInTheBuilding > 0) {
      scream();
    }
  }
}

void positive_indirect_outer_and_rhs_inner_and_lhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (someOtherCondition()) {
      if (onFire && peopleInTheBuilding > 0) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: if ( peopleInTheBuilding > 0) {
        scream();
      }
    }
  }
}

void positive_direct_inner_outer_and_rhs_and_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (peopleInTheBuilding > 0 && onFire) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: if (peopleInTheBuilding > 0 ) {
      scream();
    }
  }
}

void positive_indirect_outer_and_rhs_inner_and_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (someOtherCondition()) {
      if (peopleInTheBuilding > 0 && onFire) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: if (peopleInTheBuilding > 0 ) {
        scream();
      }
    }
  }
}

void positive_direct_outer_and_rhs_inner_or_lhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (onFire || isCollapsing()) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: {{^\ *$}}
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
  }
}

void positive_indirect_outer_and_rhs_inner_or_lhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (someOtherCondition()) {
      if (onFire || isCollapsing()) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: {{^\ *$}}
        scream();
      }
      // CHECK-FIXES: {{^\ *$}}
    }
  }
}

void positive_direct_outer_and_rhs_inner_or_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (isCollapsing() || onFire) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: {{^\ *$}}
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
  }
}

void positive_indirect_outer_and_rhs_inner_or_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (someOtherCondition()) {
      if (isCollapsing() || onFire) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: {{^\ *$}}
        scream();
      }
      // CHECK-FIXES: {{^\ *$}}
    }
  }
}

//===--- Basic Negatives --------------------------------------------------===//

void negative_direct() {
  bool onFire = isBurning();
  if (onFire) {
    tryToExtinguish(onFire);
    if (onFire) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect() {
  bool onFire = isBurning();
  if (onFire) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2() {
  bool onFire = isBurning();
  if (onFire) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_inner_and_lhs() {
  bool onFire = isBurning();
  if (onFire) {
    tryToExtinguish(onFire);
    if (onFire && peopleInTheBuilding > 0) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_inner_and_lhs() {
  bool onFire = isBurning();
  if (onFire) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (onFire && peopleInTheBuilding > 0) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_inner_and_lhs() {
  bool onFire = isBurning();
  if (onFire) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (onFire && peopleInTheBuilding > 0) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_inner_and_rhs() {
  bool onFire = isBurning();
  if (onFire) {
    tryToExtinguish(onFire);
    if (peopleInTheBuilding > 0 && onFire) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_inner_and_rhs() {
  bool onFire = isBurning();
  if (onFire) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (peopleInTheBuilding > 0 && onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_inner_and_rhs() {
  bool onFire = isBurning();
  if (onFire) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (peopleInTheBuilding > 0 && onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_inner_or_lhs() {
  bool onFire = isBurning();
  if (onFire) {
    tryToExtinguish(onFire);
    if (onFire || isCollapsing()) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_inner_or_lhs() {
  bool onFire = isBurning();
  if (onFire) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (onFire || isCollapsing()) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_inner_or_lhs() {
  bool onFire = isBurning();
  if (onFire) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (onFire || isCollapsing()) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_inner_or_rhs() {
  bool onFire = isBurning();
  if (onFire) {
    tryToExtinguish(onFire);
    if (isCollapsing() || onFire) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_inner_or_rhs() {
  bool onFire = isBurning();
  if (onFire) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (isCollapsing() || onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_inner_or_rhs() {
  bool onFire = isBurning();
  if (onFire) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (isCollapsing() || onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_outer_and_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    tryToExtinguish(onFire);
    if (onFire) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_outer_and_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_outer_and_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_outer_and_lhs_inner_and_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    tryToExtinguish(onFire);
    if (onFire && peopleInTheBuilding > 0) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_outer_and_lhs_inner_and_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (onFire && peopleInTheBuilding > 0) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_outer_and_lhs_inner_and_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (onFire && peopleInTheBuilding > 0) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_outer_and_lhs_inner_and_rhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    tryToExtinguish(onFire);
    if (peopleInTheBuilding > 0 && onFire) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_outer_and_lhs_inner_and_rhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (peopleInTheBuilding > 0 && onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_outer_and_lhs_inner_and_rhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (peopleInTheBuilding > 0 && onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_outer_and_lhs_inner_or_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    tryToExtinguish(onFire);
    if (onFire || isCollapsing()) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_outer_and_lhs_inner_or_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (onFire || isCollapsing()) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_outer_and_lhs_inner_or_lhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (onFire || isCollapsing()) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_outer_and_lhs_inner_or_rhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    tryToExtinguish(onFire);
    if (isCollapsing() || onFire) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_outer_and_lhs_inner_or_rhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (isCollapsing() || onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_outer_and_lhs_inner_or_rhs() {
  bool onFire = isBurning();
  if (onFire && fireFighters < 10) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (isCollapsing() || onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_outer_and_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    tryToExtinguish(onFire);
    if (onFire) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_outer_and_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_outer_and_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_outer_and_rhs_inner_and_lhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    tryToExtinguish(onFire);
    if (onFire && peopleInTheBuilding > 0) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_outer_and_rhs_inner_and_lhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (onFire && peopleInTheBuilding > 0) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_outer_and_rhs_inner_and_lhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (onFire && peopleInTheBuilding > 0) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_inner_outer_and_rhs_and_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    tryToExtinguish(onFire);
    if (peopleInTheBuilding > 0 && onFire) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_outer_and_rhs_inner_and_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (peopleInTheBuilding > 0 && onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_outer_and_rhs_inner_and_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (peopleInTheBuilding > 0 && onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_outer_and_rhs_inner_or_lhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    tryToExtinguish(onFire);
    if (onFire || isCollapsing()) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_outer_and_rhs_inner_or_lhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (onFire || isCollapsing()) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_outer_and_rhs_inner_or_lhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (onFire || isCollapsing()) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_direct_outer_and_rhs_inner_or_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    tryToExtinguish(onFire);
    if (isCollapsing() || onFire) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_indirect_outer_and_rhs_inner_or_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    tryToExtinguish(onFire);
    if (someOtherCondition()) {
      if (isCollapsing() || onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_indirect2_outer_and_rhs_inner_or_rhs() {
  bool onFire = isBurning();
  if (fireFighters < 10 && onFire) {
    if (someOtherCondition()) {
      tryToExtinguish(onFire);
      if (isCollapsing() || onFire) {
        // NO-MESSAGE: fire may have been extinguished
        scream();
      }
    }
  }
}

void negative_reassigned() {
  bool onFire = isBurning();
  if (onFire) {
    onFire = isReallyBurning();
    if (onFire) {
      // NO-MESSAGE: it was a false alarm then
      scream();
    }
  }
}

//===--- Special Positives ------------------------------------------------===//

// Condition variable mutated in or after the inner loop

void positive_direct_mutated_after_inner() {
  bool onFire = isBurning();
  if (onFire) {
    if (onFire) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: {{^\ *$}}
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
    tryToExtinguish(onFire);
  }
}

void positive_indirect_mutated_after_inner() {
  bool onFire = isBurning();
  if (onFire) {
    if (someOtherCondition()) {
      if (onFire) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: {{^\ *$}}
        scream();
      }
      // CHECK-FIXES: {{^\ *$}}
    }
    tryToExtinguish(onFire);
  }
}

void positive_indirect2_mutated_after_inner() {
  bool onFire = isBurning();
  if (onFire) {
    if (someOtherCondition()) {
      if (onFire) {
        // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
        // CHECK-FIXES: {{^\ *$}}
        scream();
      }
      // CHECK-FIXES: {{^\ *$}}
      tryToExtinguish(onFire);
    }
  }
}

void positive_mutated_in_inner() {
  bool onFire = isBurning();
  if (onFire) {
    if (onFire) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: {{^\ *$}}
      tryToExtinguish(onFire);
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
  }
}

void positive_or_lhs_with_side_effect() {
  bool onFire = isBurning();
  if (onFire) {
    if (callTheFD() || onFire) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: callTheFD() ;
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
  }
}

void positive_or_rhs_with_side_effect() {
  bool onFire = isBurning();
  if (onFire) {
    if (onFire || callTheFD()) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: callTheFD();
      scream();
    }
    // CHECK-FIXES: {{^\ *$}}
  }
}

// GNU Expression Statements

void doSomething();

void positive_gnu_expression_statement() {
  bool onFire = isBurning();
  if (({ doSomething(); onFire; })) {
    if (({ doSomething(); onFire; })) {
      // FIXME: Handle GNU epxression statements
      // CHECK-MESSAGES-NOT: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHCK-FIXES-NOT: doSomething();
      scream();
    }
  }
}

// Comma after Condition

void positive_comma_after_condition() {
  bool onFire = isBurning();
  if (doSomething(), onFire) {
    if (doSomething(), onFire) {
      // FIXME: Handle comma operator
      // CHECK-MESSAGES-NOT: :[[@LINE-1]]:5: warning: redundant condition 'onFire' [bugprone-redundant-branch-condition]
      // CHCK-FIXES-NOT: doSomething();
      scream();
    }
  }
}

// ExprWithCleanups doesn't crash
int positive_expr_with_cleanups() {
  class RetT {
  public:
    RetT(const int _code) : code_(_code) {}
    bool Ok() const { return code_ == 0; }
    static RetT Test(bool &_isSet) { return 0; }

  private:
    int code_;
  };

  bool isSet = false;
  if (RetT::Test(isSet).Ok() && isSet) {
    if (RetT::Test(isSet).Ok() && isSet) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'isSet' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: if (RetT::Test(isSet).Ok() ) {
    }
  }
  if (isSet) {
    if ((RetT::Test(isSet).Ok() && isSet)) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: redundant condition 'isSet' [bugprone-redundant-branch-condition]
      // CHECK-FIXES: if ((RetT::Test(isSet).Ok() )) {
    }
  }
  return 0;
}

//===--- Special Negatives ------------------------------------------------===//

// Aliasing

void negative_mutated_by_ptr() {
  bool onFire = isBurning();
  bool *firePtr = &onFire;
  if (onFire) {
    tryToExtinguish(*firePtr);
    if (onFire) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

void negative_mutated_by_ref() {
  bool onFire = isBurning();
  bool &fireRef = onFire;
  if (onFire) {
    tryToExtinguish(fireRef);
    if (onFire) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

// Volatile

void negatvie_volatile() {
  bool volatile onFire = isBurning();
  if (onFire) {
    if (onFire) {
      // NO-MESSAGE: maybe some other thread extinguished the fire
      scream();
    }
  }
}

void negative_else_branch(bool isHot) {
  bool onFire = isBurning();
  if (onFire) {
    tryPutFireOut();
  } else {
    if (isHot && onFire) {
      // NO-MESSAGE: new check is in the `else` branch
      // FIXME: handle `else` branches and negated conditions
      scream();
    }
  }
}

// GNU Expression Statements

void negative_gnu_expression_statement() {
  bool onFire = isBurning();
  if (({ doSomething(); onFire; })) {
    tryToExtinguish(onFire);
    if (({ doSomething(); onFire; })) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

// Comma after Condition

void negative_comma_after_condition() {
  bool onFire = isBurning();
  if (doSomething(), onFire) {
    tryToExtinguish(onFire);
    if (doSomething(), onFire) {
      // NO-MESSAGE: fire may have been extinguished
      scream();
    }
  }
}

//===--- Unhandled Cases --------------------------------------------------===//

void negated_in_else() {
  bool onFire = isBurning();
  if (onFire) {
    scream();
  } else {
    if (!onFire) {
      doSomething();
    }
  }
}

void equality() {
  if (peopleInTheBuilding == 1) {
    if (peopleInTheBuilding == 1) {
      doSomething();
    }
  }
}

void relational_operator() {
  if (peopleInTheBuilding > 2) {
    if (peopleInTheBuilding > 1) {
      doSomething();
    }
  }
}

void relational_operator_reversed() {
  if (peopleInTheBuilding > 1) {
    if (1 < peopleInTheBuilding) {
      doSomething();
    }
  }
}
