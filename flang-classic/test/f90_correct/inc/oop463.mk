#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop463  ########


oop463: run
	

build:  $(SRC)/oop463.f90
	-$(RM) oop463.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop463.f90 -o oop463.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop463.$(OBJX) check.$(OBJX) $(LIBS) -o oop463.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop463
	oop463.$(EXESUFFIX)

verify: ;

