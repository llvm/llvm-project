#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test he00  ########


he00: run
	

build:  $(SRC)/he00.f
	-$(RM) he00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/he00.f -o he00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) he00.$(OBJX) check.$(OBJX) $(LIBS) -o he00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test he00
	he00.$(EXESUFFIX)

verify: ;

he00.run: run

