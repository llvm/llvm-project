#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test hb00  ########


hb00: run
	

build:  $(SRC)/hb00.f
	-$(RM) hb00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/hb00.f -o hb00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) hb00.$(OBJX) check.$(OBJX) $(LIBS) -o hb00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test hb00
	hb00.$(EXESUFFIX)

verify: ;

hb00.run: run

