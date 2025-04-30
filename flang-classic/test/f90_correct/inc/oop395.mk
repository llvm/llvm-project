#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop395  ########


oop395: run
	

build:  $(SRC)/oop395.f90
	-$(RM) oop395.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop395.f90 -o oop395.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop395.$(OBJX) check.$(OBJX) $(LIBS) -o oop395.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop395
	oop395.$(EXESUFFIX)

verify: ;

