#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

$(TEST): run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/$(TEST).f08 fcheck.$(OBJX)
	-$(RM) $(TEST).$(EXESUFFIX) $(TEST).$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/$(TEST).f08 -o $(TEST).$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) $(TEST).$(OBJX) fcheck.$(OBJX) $(LIBS) -o $(TEST).$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test $(TEST)
	$(TEST).$(EXESUFFIX)

verify: ;

