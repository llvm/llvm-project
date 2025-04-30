#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test hf00  ########


hf00: run
	

build:  $(SRC)/hf00.f
	-$(RM) hf00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/hf00.f -o hf00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) hf00.$(OBJX) check.$(OBJX) $(LIBS) -o hf00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test hf00
	hf00.$(EXESUFFIX)

verify: ;

hf00.run: run

