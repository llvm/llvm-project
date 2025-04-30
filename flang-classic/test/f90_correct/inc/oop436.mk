#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop436  ########


oop436: run
	

build:  $(SRC)/oop436.f90
	-$(RM) oop436.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop436.f90 -o oop436.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop436.$(OBJX) check.$(OBJX) $(LIBS) -o oop436.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop436
	oop436.$(EXESUFFIX)

verify: ;

