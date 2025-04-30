#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
do13: do13.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
do13.$(OBJX): $(SRC)/do13.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/do13.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) do13.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: do13
run: ;
