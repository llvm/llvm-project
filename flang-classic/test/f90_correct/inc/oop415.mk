#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop415  ########


oop415: run
	

build:  $(SRC)/oop415.f90
	-$(RM) oop415.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop415.f90 -o oop415.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop415.$(OBJX) check.$(OBJX) $(LIBS) -o oop415.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop415
	oop415.$(EXESUFFIX)

verify: ;

