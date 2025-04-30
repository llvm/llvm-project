#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre08  ########


pre08: run
FFLAGS += -mp -Mpreprocess
	

build:  $(SRC)/pre08.f
	-$(RM) pre08.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre08.f -o pre08.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre08.$(OBJX) check.$(OBJX) $(LIBS) -o pre08.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre08
	pre08.$(EXESUFFIX)

verify: ;

pre08.run: run

