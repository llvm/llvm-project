#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qdim  ########


qdim: run
	

build:  $(SRC)/qdim.f08
	-$(RM) qdim.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qdim.f08 -o qdim.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qdim.$(OBJX) check.$(OBJX) $(LIBS) -o qdim.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qdim
	qdim.$(EXESUFFIX)

verify: ;

qdim.run: run

