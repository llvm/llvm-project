#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre21  ########


pre21: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre21.f90
	-$(RM) pre21.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre21.f90 -o pre21.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre21.$(OBJX) check.$(OBJX) $(LIBS) -o pre21.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre21
	pre21.$(EXESUFFIX)

verify: ;

pre21.run: run

