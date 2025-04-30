#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

$(TEST): run
	

build:  $(SRC)/$(TEST).f90
	-$(RM) $(TEST).$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/$(TEST).f90 -o $(TEST).$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) $(TEST).$(OBJX) check.$(OBJX) $(LIBS) -o $(TEST).$(EXESUFFIX)
	-$(FC) -c -i8 $(FFLAGS) $(LDFLAGS) $(SRC)/$(TEST).f90 -o $(TEST).$(OBJX).i8
	-$(FC) $(FFLAGS) $(LDFLAGS) $(TEST).$(OBJX).i8 check.$(OBJX) $(LIBS) -o $(TEST).$(EXESUFFIX).i8


run: 
	@echo ------------------------------------ executing test $(TEST)
	$(TEST).$(EXESUFFIX)
	$(TEST).$(EXESUFFIX).i8

verify: ;

