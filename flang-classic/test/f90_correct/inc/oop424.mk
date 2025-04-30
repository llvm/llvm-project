#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop424  ########


oop424: run
	

build:  $(SRC)/oop424.f90
	-$(RM) oop424.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop424.f90 -o oop424.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop424.$(OBJX) check.$(OBJX) $(LIBS) -o oop424.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop424
	oop424.$(EXESUFFIX)

verify: ;

