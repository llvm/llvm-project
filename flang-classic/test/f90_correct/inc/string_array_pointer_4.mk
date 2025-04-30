#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

$(TEST): run

build: $(SRC)/$(TEST).f90
	-$(RM) $(TEST).$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/$(TEST).f90 -o $(TEST).$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test $(TEST)
	$(TEST).$(EXESUFFIX)

verify: ;
