#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test slogtoq  ########


qp93: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp93.f08 fcheck.$(OBJX)
	-$(RM) qp93.$(EXESUFFIX) qp93.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp93.f08 -o qp93.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp93.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp93.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp93
	qp93.$(EXESUFFIX)

verify: ;

qp93.run: run

