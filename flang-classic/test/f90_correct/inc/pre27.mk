#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre27  ########


pre27: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre27.f90
	-$(RM) pre27.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre27.f90 -o pre27.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre27.$(OBJX) check.$(OBJX) $(LIBS) -o pre27.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre27
	pre27.$(EXESUFFIX)

verify: ;

pre27.run: run

