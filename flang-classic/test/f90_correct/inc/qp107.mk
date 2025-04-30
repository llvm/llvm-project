#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test holltoq  ########


qp107: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp107.f08 fcheck.$(OBJX)
	-$(RM) qp107.$(EXESUFFIX) qp107.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp107.f08 -o qp107.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp107.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp107.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp107
	qp107.$(EXESUFFIX)

verify: ;

qp107.run: run

