// Copyright (c) 2019, Arm Ltd.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Reminder: no warranty with this program. I recommend using a fresh checkout.

// Compile with:
//   clang++ -Wall -Werror -O2 flatten.cpp -lgit2
// Run with f18 in PWD or argv[1]:
//   time ./a.out ~/.local/src/github.com/flang-compiler/f18/

// To get a rewritten history, do this:
//
//   sudo apt install -y libgit2-dev # or equivalent
//   git clone https://github.com/flang-compiler/f18
//   git remote add llvm-project https://github.com/llvm/llvm-project
//   git fetch llvm-project
//   clang++ -DREPLACE_REFERENCES -Wall -Werror -O2 flatten.cpp -lgit2
//   ./a.out

// Inputs:
// * a ref called origin/master, representing f18 history
// * (optionally) a ref called llvm-project/master, representing llvm upstream
// * (optionally) branches called rebase-{12 digit merge sha}, representing the
//   manual rebase of tricky cases.
//
// Outputs:
// * A branch called rewritten-history-v2, with a linearized f18 history.
// * A branch called rewritten-history-v2-llvm-project-merge, representing the
//   renaming of the project under /flang/ and taking llvm-project/master as the
//   new parent for the (original) root commit.
//
// This program is meant to be idempotent and should not write to the working
// directory, it simply takes refs as input and produces them as output.

// Key concepts:
//
// * The checkout that git gives you for a commit is called a "tree", which is
//   determined by a recursive checksum of the directory structure. If two
//   commits have the same tree ("treesame"), then they are by definition
//   equivalent when you check them out.
//
// * Lineage of the "master" branch is taken by following the first parents of
//   each commit. To see this in git log, run `git log --first-parent`. This
//   effectively ignores the second-parent history (i.e. commits that happened
//   on branches).
//
// * By construction it is arranged that the trees of the first-parent history
//   are preserved. This means "the code on the master branch is the same before
//   and after rewrite".
//
// * Preserving the non-first-parent commits is trickier, and requires a rebase.
//
// * If nothing changed on the master branch during a feature branch, a rebase
//   will not change the trees of the feature branch, so trees of those commits
//   will still be the same. It's like rewriting the merge as a fast-forward.
//
// * However, if something happened on the master branch during the feature
//   branch, then a rebase *must* create new trees. This implies code which
//   might not build. As an example, imagine a case where a class is renamed on
//   master, and the old name is used in the feature branch (until it's fixed at
//   some point by the time it is merged).
//
// * By the end of the rebase, we assert that the trees are the same as those
//   merged into master. So code in the middle of the rebased feature branch may
//   not build, but at least the overall result of the feature branch will be as
//   good as master was. Thankfully this is relatively rare.
//
// * If a branch exists called rebase-{sha of merge commit}, that branch is
//   substituted in place of the merge commit. This allows manually rebasing
//   tricky merges.
//
// * For the non-treesame, we can take a second-order diff (diff-of-diff)
//   comparing those commits before and after rewrite, and ensure that only line
//   numbers and context changed. This is almost totally the case.

// Using the following script, it is possible to see whether non-TREESAME
// patches still have the same diff, modulo blank lines, by taking a
// second-order diff.
//
// git log --grep=TREESAME --invert-grep  --format="%h %(trailers:key=Original-commit)" rewritten-history-v2 |
//   sed -n 's|Original-commit: flang-compiler/f18@||p' |
//   while read NEW ORIG
//   do
//     echo ORIG NEW: $ORIG $NEW
//     git show $ORIG > a
//     git show $NEW > b
//     sed -r -i \
//       -e 's/@@ .* @@/@@ Numbers @@/g' \
//       -e '/^(commit|index) .*/d' \
//       -e '/Original-commit.*/d' \
//       -e '/^\s$/d' \
//        a b
//     git diff --color --no-index a b
//   done |& less -SR

#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>

#ifndef NO_REPLACE_REFERENCES
#include <regex>
#endif

#include <sys/stat.h>
#include <sys/types.h>

#include <git2.h>
#include <git2/sys/commit.h>

void check(int error, const char *message, const char *extra) {
    const git_error *err;
    const char *msg = "", *spacer = "";

    if (!error)
        return;

    if ((err = giterr_last()) != NULL && err->message != NULL) {
        msg = err->message;
        spacer = " - ";
    }

    if (extra)
        fprintf(stderr, "%s '%s' [%d]%s%s\n", message, extra, error, spacer, msg);
    else
        fprintf(stderr, "%s [%d]%s%s\n", message, error, spacer, msg);

    exit(1);
}

int n_conflicts = 0, n_discards = 0;

// Copy src string to dst string, rewriting issue references.
char *rewrite_issue_references(char *dst, const char *src) {
#ifndef NO_REPLACE_REFERENCES
    const char *src_end = src + strlen(src);
    // return src_end;
    char *new_end = std::regex_replace(
        dst, src, src_end,
        std::regex("(^|\\b[^a-zA-Z0-9]+)(#[0-9]+)\\b"),
        "$1flang-compiler/f18$2");
    *new_end = '\0';
    return new_end;
#else
    return stpcpy(dst, src);
#endif
}

// test_rewrite_issue_references runs some test cases thorugh the string
// replacement machinery and aborts if anything is awry.
void test_rewrite_issue_references() {
    #ifdef NO_REPLACE_REFERENCES
        return;
    #endif
    struct { const char *input, *want; } tests[] = {
        {"foo#123", "foo#123"},
        {"Test #123bar", "Test #123bar"},
        // Special case.
        // {"commit message #123", "commit message #123"},

        {"#123", "flang-compiler/f18#123"},
        {"Test #123", "Test flang-compiler/f18#123"},
        {"Test #123", "Test flang-compiler/f18#123"},
        {"Test (#123)", "Test (flang-compiler/f18#123)"},
    };

    bool fail = false;
    for (const auto test : tests) {
        char *x = (char*)malloc(1024);
        const char *new_end = rewrite_issue_references(x, test.input);
        if (strcmp(x, test.want)) {
            fprintf(stderr, "Got : %s\n", x);
            fprintf(stderr, "Want: %s\n", test.want);
            fail = true;
        }
        if (new_end != x + strlen(x)) {
            abort();
        }
        (void)new_end;
        free((void*)x);
    }
    if (fail)
        abort();
}

static const char mergemsg_prefix[] = "Merge pull request #";

// has_merge_pr_prefix returns true if the commit message begins "Merge pull
// request #".
bool has_merge_pr_prefix(const char* msg) {
    int len = sizeof(mergemsg_prefix)-1;
    if (strlen(msg) < len)
        len = strlen(msg);
    return !strncmp(mergemsg_prefix, msg, len);
}

// tweak_commit_message
// Prepend [flang-compiler/f18#PRNUM]
// Append "Original-commit", "Reviewed-on" and "Tree-same-pre-rewrite".
//
// Allocates a new commit message. Return value must be freed.
// The Reviewed-on trailer URL is determined by "Merge pull request #(number)",
// if present.
char *tweak_commit_message(git_commit *orig_commit, git_commit *orig_merge, const git_oid *new_tree) {
    const char *orig_msg = git_commit_message_raw(orig_commit);
    const char *prnum = NULL, *prnum_end = NULL;

    // If the message indicates a PR, store in prnum.
    if (orig_merge != NULL && has_merge_pr_prefix(git_commit_message(orig_merge))) {
        const char *mergemsg = git_commit_message_raw(orig_merge);
        prnum = mergemsg + sizeof(mergemsg_prefix) - 1;
        prnum_end = strchr(prnum, ' ');
    }

    #ifndef NO_REPLACE_REFERENCES
    // Match "foo bar baz (#123)", which is the convention for "Squash" commit
    // merges on GitHub.
    static std::regex prnum_re("^(.*\\(#)([0-9]+)\\)$");
    std::cmatch match;
    if (std::regex_match(git_commit_summary(orig_merge), match, prnum_re)) {
        const char *summary = git_commit_summary(orig_merge);
        prnum = summary + match.length(1);
        prnum_end = prnum + match.length(2);
    }
    #endif

    // Gratuitous space for appending things.
    const ssize_t extra_space = 102400;
    ssize_t size = strlen(orig_msg) + extra_space;
    char *newmsg_start = (char*)malloc(size);
    char *newmsg_end = newmsg_start + (size);
    char *newmsg = newmsg_start; // Pointer tracks the current write position.
    newmsg[0] = 0;

    // Set to leave message unmodified except for Original-commit, useful for
    // verifying second-order diffs.
    const bool use_original_message = false;
    if (use_original_message) {
        // These are here to indicate if the checkouts are the same as a commit and/or a merge.
        if (git_oid_equal(git_commit_tree_id(orig_merge), new_tree)) {
            newmsg = stpncpy(newmsg, "[TREESAME master] ", newmsg_end - newmsg);
        } else if (git_oid_equal(git_commit_tree_id(orig_commit), new_tree)) {
            newmsg = stpncpy(newmsg, "[TREESAME commit] ", newmsg_end - newmsg);
        }

        newmsg = stpcpy(newmsg, orig_msg);

        // From here on out, append trailer headers.
        char buf[GIT_OID_HEXSZ+1] = {};
        newmsg = stpncpy(newmsg, "\n\nOriginal-commit: flang-compiler/f18@", newmsg_end - newmsg);
        git_oid_fmt(buf, git_commit_id(orig_commit));
        newmsg = stpncpy(newmsg, buf, newmsg_end - newmsg);
        newmsg = stpncpy(newmsg, "\n", newmsg_end - newmsg);
        return newmsg_start;
    }

    // Prepend [Flang] tag.
    newmsg = stpncpy(newmsg, "[Flang] ", newmsg_end - newmsg);

    // Paste in the original message, rewriting references #123 => flang-compiler/f18#123
    newmsg = rewrite_issue_references(newmsg, orig_msg);

    // If there is a newline at the end, remove it; subsequent insertion of the
    // Original-commit header will always insert it. This ensures consistent
    // spacing before the header.
    while (newmsg[-1] == '\n') {
        newmsg[-1] = 0;
        newmsg--;
    }

    // From here on out, append trailer headers.
    char buf[GIT_OID_HEXSZ+1] = {};
    newmsg = stpncpy(newmsg, "\n\nOriginal-commit: flang-compiler/f18@", newmsg_end - newmsg);
    git_oid_fmt(buf, git_commit_id(orig_commit));
    newmsg = stpncpy(newmsg, buf, newmsg_end - newmsg);
    newmsg = stpncpy(newmsg, "\n", newmsg_end - newmsg);

    if (prnum != NULL) {
        newmsg = stpncpy(newmsg, "Reviewed-on: https://github.com/flang-compiler/f18/pull/", newmsg_end - newmsg);
        newmsg = stpncpy(newmsg, prnum, prnum_end - prnum);
        newmsg = stpncpy(newmsg, "\n", newmsg_end - newmsg);
    }

    if (!git_oid_equal(git_commit_tree_id(orig_merge), new_tree)) {
        // If this is present, then the contents of the tree are identical pre-
        // and post- merge. If it is not present, then the patch was rebased.
        newmsg = stpncpy(newmsg, "Tree-same-pre-rewrite: false\n", newmsg_end - newmsg);
    }

    return newmsg_start;
}

// insert_flang_directory sets new_root to a newly created tree with one entry
// in it: /flang/, which points at orig_root.
void insert_flang_directory(git_repository *repo, git_oid *new_root, const git_oid *orig_root) {
    git_treebuilder *tb;
    check(git_treebuilder_new(&tb, repo, NULL), "git_treebuilder_new", NULL);
    const git_tree_entry *te;
    git_treebuilder_insert(&te, tb, "flang", orig_root, GIT_FILEMODE_TREE);
    git_treebuilder_write(new_root, tb);
    git_treebuilder_free(tb);
}

// count_branch_commits counts the number of on-branch (non-merge) commits in
// the given merge.
int count_branch_commits(git_commit *merge) {
    git_revwalk *walk;
    check(git_revwalk_new(&walk, git_commit_owner(merge)), "git_revwalk_new", NULL);
    check(git_revwalk_hide(walk, git_commit_parent_id(merge, 0)), "git_revwalk_hide", NULL);
    check(git_revwalk_push(walk, git_commit_parent_id(merge, 1)), "git_revwalk_push", NULL);

    git_oid commit_oid;
    int n = 0;
    while (!git_revwalk_next(&commit_oid, walk))
        n++;

    git_revwalk_free(walk);
    return n;
}

// tree_for_commit grabs the git_oid pointing to the tree for a given commit_id.
git_oid tree_for_commit(git_repository *repo, const git_oid *commit_id) {
    git_commit *c;
    check(git_commit_lookup(&c, repo, commit_id), "git_commit_lookup", NULL);
    git_oid tree_id;
    git_oid_cpy(&tree_id, git_commit_tree_id(c));
    git_commit_free(c);
    // git_commit_
    return tree_id;
}

// generate_authortime_to_commit_map walks the commits on the second-parent
// history of the given `merge`, computing a mapping from the author time to the
// original commit id. Since this is scoped to feature-branch commits, there are
// not likely to be collisions.
void generate_authortime_to_commit_map(std::map<git_time_t, git_oid> &authortime_to_commit, git_commit *merge) {
    git_repository *repo = git_commit_owner(merge);
    git_revwalk *walk;
    check(git_revwalk_new(&walk, git_commit_owner(merge)), "git_revwalk_new", NULL);
    check(git_revwalk_hide(walk, git_commit_parent_id(merge, 0)), "git_revwalk_hide", NULL);
    check(git_revwalk_push(walk, git_commit_parent_id(merge, 1)), "git_revwalk_push", NULL);

    // Only walk first parent history on the grounds that most of those which
    // introduce commits not-already-on-mainline are accidental merges of
    // rebases, duplicating patches in history. Where patches are missed, they
    // won't have an entry in the authortime_to_commit.
    git_revwalk_simplify_first_parent(walk);

    git_oid commit_id;
    while (!git_revwalk_next(&commit_id, walk)) {
        git_commit *c;
        check(git_commit_lookup(&c, repo, &commit_id), "git_commit_lookup", NULL);
        int when = git_commit_author(c)->when.time;

        if (authortime_to_commit.count(when) != 0) {
            char buf[GIT_OID_HEXSZ+1] = {};
            git_oid_nfmt(buf, 12, &commit_id);
            char buf1[GIT_OID_HEXSZ+1] = {};
            git_oid_nfmt(buf1, 12, git_commit_id(merge));
            char buf2[GIT_OID_HEXSZ+1] = {};
            git_oid_nfmt(buf2, 12, &authortime_to_commit[when]);
            printf("Hit duplicate commit considering %s "
                   "(merge %s, duplicate %s)\n", buf, buf1, buf2);
            // Duplicate author times. Need another strategy.
            abort();
        }
        authortime_to_commit[when] = commit_id;
        git_commit_free(c);
    }

    git_revwalk_free(walk);
}

// try_rebase attempts to rebase orig_merge onto the new history.
// It returns true if the rebase succeeds without conflicts, and false otherwise.
// On success, new_head is set to the tip of the rebase.
bool try_rebase(git_oid **new_head, git_commit *orig_merge) {
    git_repository *repo = git_commit_owner(orig_merge);
    const git_oid *p0 = git_commit_parent_id(orig_merge, 0);
    const git_oid *p1 = git_commit_parent_id(orig_merge, 1);

    git_annotated_commit *p0a, *p1a, *new_heada;
    check(git_annotated_commit_lookup(&p0a, repo, p0), "git_annotated_commit_lookup p0", NULL);
    check(git_annotated_commit_lookup(&p1a, repo, p1), "git_annotated_commit_lookup p1", NULL);
    check(git_annotated_commit_lookup(&new_heada, repo, *new_head), "git_annotated_commit_lookup new_head", NULL);

    char buf[] = "refs/heads/rebase-0123456789ab\0";
    git_oid_nfmt(buf+sizeof("refs/heads/rebase-")-1, 12, git_commit_id(orig_merge));

    bool using_manual_rebase = false;

    // Look for a branch called rebase-[12 digit SHA]. If it exists and is
    // tree-same to the merge, treat it as the branch we're trying to rebase.
    git_reference *manual_rebase;
    int err = git_reference_lookup(&manual_rebase, repo, buf);
    switch (err) {
    case 0: { // Reference found.
        const git_oid manual_tree = tree_for_commit(repo, git_reference_target(manual_rebase));

        if (0 == git_oid_cmp(
                git_reference_target(manual_rebase),
                git_commit_id(orig_merge))) {
            printf("Skip %s because it's pointing at the merge.\n", buf);
            goto manual_rebase_unusable;
        }
        if (0 != git_oid_cmp(&manual_tree, git_commit_tree_id(orig_merge))) {
            printf("Skip %s because the tip of the rebase is not "
                   "treesame to the merge commit\n", buf);
            goto manual_rebase_unusable;
        }
        printf("Using manual rebase branch %s\n", buf);
        using_manual_rebase = true;

        // Update p1a, the commits being rebased, to point at the branch.
        // Then rebase, and this shouldn't result in any conflicts.
        git_annotated_commit_free(p1a);
        git_annotated_commit_lookup(&p1a, repo, git_reference_target(manual_rebase));

    manual_rebase_unusable:
        git_reference_free(manual_rebase);

        break;
    }
    case GIT_ENOTFOUND:
        // printf("Rebase branch %s not found.\n", buf);
        break;
    default:
        check(err, "git_reference_lookup rebase-...", NULL);
    }

    git_rebase_options rb_opts;
    check(git_rebase_init_options(&rb_opts, GIT_REBASE_OPTIONS_VERSION), "git_rebase_init_options", NULL);
    rb_opts.inmemory = 1;
    rb_opts.merge_options.flags = GIT_MERGE_FIND_RENAMES;
    rb_opts.merge_options.rename_threshold = 50;

    git_rebase *rb;
    check(git_rebase_init(&rb, repo, p1a, p0a, new_heada, &rb_opts), "git_rebase_init", NULL);

    bool is_success = true; // becomes false if conflicts encountered.
    bool committed_at_least_one_patch = false;
    git_oid rebase_tip_id;
    git_oid_cpy(&rebase_tip_id, *new_head);

    std::map<git_time_t, git_oid> authortime_to_commit;
    if (using_manual_rebase) {
        generate_authortime_to_commit_map(authortime_to_commit, orig_merge);
    }

    // Loop over each patch in the rebase, committing it.
    git_rebase_operation *op;
    while (!git_rebase_next(&op, rb)) {
        git_index *idx;
        check(git_rebase_inmemory_index(&idx, rb), "git_rebase_inmemory_index", NULL);
        if (git_index_has_conflicts(idx)) {
            // Conflicting case. Print a useful message.
            char buf_patch[GIT_OID_HEXSZ+1] = {};
            char buf_merge[GIT_OID_HEXSZ+1] = {};
            git_oid_nfmt(buf_patch, 12, &op->id);
            git_oid_nfmt(buf_merge, 12, git_commit_id(orig_merge));

            int discarded = count_branch_commits(orig_merge);
            printf("Conflicts encountered; patch=%s merge=%s - discarding %d commits\n", buf_patch, buf_merge, discarded);
            printf("  M=%s; git checkout -B rebase-${M} ${M}^2; git rebase ${M}^1\n", buf_merge);

            n_conflicts++;
            n_discards += discarded;

            git_index_free(idx);
            is_success = false;
            // If conflicts are found, abort, fall back to taking the merge
            // commit.
            break;
        }
        git_index_free(idx);

        git_commit *orig_commit;
        check(git_commit_lookup(&orig_commit, repo, &op->id), "git_commit_lookup", NULL);

        // Generate the new tree now (as opposed to within git_rebase_commit) so that it can be used for TREESAME
        // diagnostics in the commit message.
        git_oid new_tree;
        check(git_index_write_tree_to(&new_tree, idx, repo), "git_index_write_tree_to", NULL);

        if (using_manual_rebase) {
            // If in a manual rebase, need to lookup original patch.
            // Use the author timestamp as a heuristic for patch equality.
            const git_time_t when = git_commit_author(orig_commit)->when.time;
            git_oid pre_rebase_commit_id = {};
            if (when == 1518039228) { // Wed Feb 7 13:33:48 2018 -0800
                // Hack for a single special case, a commit which was merged.
                check(git_oid_fromstr(&pre_rebase_commit_id, "044148ead21f18e16716d5bc30819525c79065d0"), "git_oid_fromstr", NULL);
            } else if (authortime_to_commit.count(when) == 0) {
                char buf[GIT_OID_HEXSZ+1] = {};
                git_oid_nfmt(buf, 12, &op->id);
                printf("Unable to find original commit for manual "
                       "rebase: %s\n", buf);
                git_oid_nfmt(buf, 12, git_commit_id(orig_merge));
                printf("  Merge: %s\n", buf);
                abort();
            } else {
                pre_rebase_commit_id = authortime_to_commit[when];
            }

            // Replace orig_commit (the rebased commit in this context) with the
            // true original commit, so that the commit cross-reference
            // correctly reflects a commit which exists in the f18 repository.
            git_commit_free(orig_commit);
            check(git_commit_lookup(&orig_commit, repo, &pre_rebase_commit_id), "git_commit_lookup", NULL);
        }

        const char *msg = tweak_commit_message(orig_commit, orig_merge, &new_tree);

        int err = git_rebase_commit(
            &rebase_tip_id,
            rb,
            NULL,
            // Take the committer information from the merge commit if manually rebased.
            using_manual_rebase ? git_commit_committer(orig_merge): git_commit_committer(orig_commit),
            NULL,
            msg
        );
        free((void*)msg);

        git_commit_free(orig_commit);
        if (err == GIT_EAPPLIED) {
            // Applying the patch results in the same tree, so the patch is
            // empty.
            char buf_patch[GIT_OID_HEXSZ+1] = {};
            char buf_merge[GIT_OID_HEXSZ+1] = {};
            git_oid_nfmt(buf_patch, 12, &op->id);
            git_oid_nfmt(buf_merge, 12, git_commit_id(orig_merge));
            printf("Patch already exists in history; patch=%s merge=%s\n", buf_patch, buf_merge);
            continue;
        }
        check(err, "git_rebase_commit", NULL);
        committed_at_least_one_patch = true;
    }

    if (is_success && committed_at_least_one_patch) {
        // Update the growing new_head to point at our new rebase tip.
        git_oid_cpy(*new_head, &rebase_tip_id);
    }

    git_rebase_abort(rb);
    git_rebase_free(rb);

    git_annotated_commit_free(p0a);
    git_annotated_commit_free(p1a);
    git_annotated_commit_free(new_heada);

    return is_success;
}

// merge_llvm_project_tree generates a new root tree combining the llvm project
// tree and the given new_tree_id. new_tree_id is updated to point at the new tree.
void merge_llvm_project_tree(
    git_oid *new_tree_id,
    const git_oid *orig_tree,
    const git_tree *llvm_project_tree) {

    git_repository *repo = git_tree_owner(llvm_project_tree);

    git_tree *flang_tree;
    check(git_tree_lookup(&flang_tree, repo, orig_tree), "git_tree_lookup", NULL);
    const git_oid *flang_dir_tree_id = git_tree_entry_id(git_tree_entry_byname(flang_tree, "flang"));

    // Effectively merges the flang/ directory into the llvm project tree.
    git_treebuilder *tb;
    check(git_treebuilder_new(&tb, repo, llvm_project_tree), "git_treebuilder_new", NULL);
    check(git_treebuilder_insert(NULL, tb, "flang", flang_dir_tree_id, GIT_FILEMODE_TREE), "git_treebuilder_insert", NULL);
    check(git_treebuilder_write(new_tree_id, tb), "git_treebuilder_write", NULL);
    git_treebuilder_free(tb);

    git_tree_free(flang_tree);
}

// generate_squash_message generates a commit message for merges which have been
// squashed.
void generate_squash_message(char **newmsg, git_commit *merge_commit) {
    std::stringstream s;

    // Start the message with the existing rewritten message.
    s << *newmsg;

    s << "\nDue to a conflicting rebase during the linearizing of "
         "flang-compiler/f18, this commit squashes a number of "
         "other commits:\n\n";

    git_revwalk *walk;
    check(git_revwalk_new(&walk, git_commit_owner(merge_commit)), "allocate git_revwalk", NULL);
    git_revwalk_simplify_first_parent(walk);
    git_revwalk_sorting(walk, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE);
    check(git_revwalk_push(walk, git_commit_parent_id(merge_commit, 1)), "git_revwalk_push", NULL);
    check(git_revwalk_hide(walk, git_commit_parent_id(merge_commit, 0)), "git_revwalk_hide", NULL);

    git_oid commit_id;
    while (!git_revwalk_next(&commit_id, walk)) {
        char buf[GIT_OID_HEXSZ+1] = {};
        git_oid_fmt(buf, &commit_id);

        git_commit *c;
        check(git_commit_lookup(&c, git_commit_owner(merge_commit), &commit_id), "git_commit_lookup", NULL);

        s << "flang-compiler/f18@" << buf << " " << git_commit_summary(c) << "\n";
        git_commit_free(c);

    }

    git_revwalk_free(walk);

    // Replace newmsg with the squashed msg.
    auto result = s.str();
    char *squashmsg = (char*)malloc(result.size()+1);
    squashmsg[result.size()] = 0;
    strncpy(squashmsg, result.c_str(), result.size());
    free(*newmsg);
    *newmsg = squashmsg;
}

int main(int argc, char* argv[]) {
    test_rewrite_issue_references();

    git_libgit2_init();

    const char *repo_path = ".";
    if (argc > 1)
        repo_path = argv[1];

    git_repository *repo;
    int error = git_repository_open(&repo, repo_path);
    if (error < 0) {
        fprintf(stderr, "Could not open repository: %s\n", giterr_last()->message);
        exit(1);
    }

    // Walk commits in reverse topological order starting from origin/master.
    git_revwalk *walk;
    check(git_revwalk_new(&walk, repo), "allocate git_revwalk", NULL);
    git_revwalk_simplify_first_parent(walk);
    git_revwalk_sorting(walk, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE);

    check(git_revwalk_push_ref(walk, "refs/remotes/origin/master"), "git_revwalk_push_head", NULL);
    // check(git_revwalk_push_ref(walk, "refs/heads/flatten-top"), "git_revwalk_push_ref", NULL);
    // check(git_revwalk_hide_ref(walk, "refs/heads/flatten-bottom"), "git_revwalk_hide_ref", NULL);

    bool is_root = true; // First commit has no parents.
    git_oid old_head = {};
    git_oid *new_head = NULL;
    git_oid new_commit_id = {};

    // For each commit in the first-parent lineage of the original history:
    //
    //   1. Take non-merge commits as they were.
    //   2. Attempt to rebase second-parent of merge commits onto first-parent.
    //      2a. Otherwise, squash them.
    //
    // Merge commits are preserved as empty commits.
    while (!git_revwalk_next(&old_head, walk)) {
        git_commit *c;
        check(git_commit_lookup(&c, repo, &old_head), "git_commit_lookup", NULL);

        // Prettify the commit message - rewrite references, add trailer headers.
        char *newmsg = tweak_commit_message(c, c, git_commit_tree_id(c));

        switch (git_commit_parentcount(c)) {
        default:
            fprintf(stderr, "Unexpected number of parents.\n");
            exit(5);

        case 2: {
            if (is_root) {
                // root commit cannot be rebased. Squash instead.
                // (only happens if using a restricted commit range)
                break;
            }
            if (try_rebase(&new_head, c)) {
                // Rebase succeeded. Now ensure that at the end of the rebase,
                // the tree state is the same as if the merge had been done.
                git_oid old_tree = tree_for_commit(repo, &old_head);
                git_oid new_tree = tree_for_commit(repo, new_head);
                if (!git_oid_equal(&old_tree, &new_tree)) {
                    char buf_old_head[GIT_OID_HEXSZ+1] = {};
                    char buf_new_head[GIT_OID_HEXSZ+1] = {};
                    git_oid_nfmt(buf_old_head, 12, &old_head);
                    git_oid_nfmt(buf_new_head, 12, new_head);

                    fprintf(stderr, "commits do not have the same tree: (old, "
                                    "new) = %s %s", buf_old_head, buf_new_head);
                    exit(6);
                }

                // Create an empty commit for the merge.
                check(git_commit_create_from_ids(
                    &new_commit_id,
                    repo,
                    NULL,
                    git_commit_author(c),
                    git_commit_committer(c),
                    git_commit_message_encoding(c),
                    newmsg,
                    &new_tree,
                    is_root ? 0 : 1,
                    (const git_oid**)(&new_head)
                ), "git_commit_create_from_ids", NULL);
                new_head = &new_commit_id;
                is_root = false;

                // Rebase succeeded, new_head updated. Keep going...
                goto next_patch;
            }

            generate_squash_message(&newmsg, c);
        }

        // These are non-merge commits on the first-parent history.
        // Take them as-is.
        case 0: case 1: ;
        }

        // Create a new commit.
        check(git_commit_create_from_ids(
            &new_commit_id,
            repo,
            NULL,
            git_commit_author(c),
            git_commit_committer(c),
            git_commit_message_encoding(c),
            newmsg,
            git_commit_tree_id(c),
            is_root ? 0 : 1,
            (const git_oid**)(&new_head)
        ), "git_commit_create_from_ids", NULL);
        new_head = &new_commit_id;
        is_root = false;

    next_patch:
        free((void*)newmsg);
        git_commit_free(c);
    }

    // First pass now done. Move the directory in a second pass, and re-parent
    // onto llvm-project if it is available.

    char buf[GIT_OID_HEXSZ+1] = {};
    git_oid_nfmt(buf, 12, new_head);
    printf("\nConflicts encountered: %d, discarding %d commits\n", n_conflicts, n_discards);
    printf("Done; rewritten-history-v4 => %s\n", buf);

    git_reference *ref;
    check(git_reference_create(
            &ref,
            repo,
            "refs/heads/rewritten-history-v4",
            new_head,
            1,
            "flatten.cpp update"
        ),
        "git_reference_create", NULL);
    git_reference_free(ref);

    git_revwalk_reset(walk);

    // Now rename everything under flang/.
    printf("Inserting /flang/...\n");
    {
        git_oid new_commit_id;
        bool is_root = true; // First commit has no parents.
        git_oid *new_head_renamed = NULL;

        git_revwalk_sorting(walk, GIT_SORT_TOPOLOGICAL | GIT_SORT_REVERSE);
        check(git_revwalk_push(walk, new_head), "git_revwalk_push_head", NULL);

        // See if the upstream is available at llvm-project/master. If it is,
        // we'll write the history into there, and use the LLVM project head as
        // the root commit.
        git_oid llvm_project_head = {};
        git_tree *llvm_project_tree;
        int err = git_reference_name_to_id(&llvm_project_head, repo, "refs/remotes/llvm-project/master");
        bool have_llvm_project = err == 0;

        if (!have_llvm_project) {
            fprintf(stderr, "Require llvm-project/master ref to exist before proceeding. Add llvm-project as a remote and fetch it.\n");
            exit(2);
        }

        git_oid_nfmt(buf, 12, &llvm_project_head);
        printf("Rewriting history on top of llvm-project@%s...\n", buf);

        // Disabled since the merged MLIR root commit has zero parents.
        // Take the same approach to be consistent (= false).
        const bool use_llvm_project_head_as_root = false;
        if (use_llvm_project_head_as_root) {
            new_head_renamed = &llvm_project_head;
            is_root = false;
        }

        // Grab the llvm_project_tree.
        git_commit *c;
        check(git_commit_lookup(&c, repo, &llvm_project_head), "git_commit_lookup", NULL);
        check(git_commit_tree(&llvm_project_tree, c), "git_commit_tree", NULL);
        git_commit_free(c);

        git_oid new_tree;

        // For each commit, rewrite its tree.
        while (!git_revwalk_next(&old_head, walk)) {
            git_commit *c;
            check(git_commit_lookup(&c, repo, &old_head), "git_commit_lookup", NULL);

            insert_flang_directory(repo, &new_tree, git_commit_tree_id(c));

            check(git_commit_create_from_ids(
                &new_commit_id,
                repo,
                NULL,
                git_commit_author(c),
                git_commit_committer(c),
                git_commit_message_encoding(c),
                git_commit_message_raw(c),
                &new_tree,
                is_root ? 0 : 1,
                (const git_oid**)(&new_head_renamed)
            ), "git_commit_create_from_ids", NULL);
            new_head_renamed = &new_commit_id;
            is_root = false;

            git_commit_free(c);
        }

        git_signature *merge_commit_author;
        check(git_signature_default(&merge_commit_author, repo), "git_signature_default", NULL);

        const char *merge_message =
            "[Flang] Merge flang-compiler/f18\n"
            "\n"
            "This is the initial merge of flang-compiler, which is done in this way\n"
            "principally to preserve the history and git-blame, without generating a large\n"
            "number of commits on the first-parent history of LLVM.\n"
            "\n"
            "If you don't care about the flang history during a bisect remember that you can\n"
            "supply paths to git-bisect, e.g. `git bisect start clang llvm`.\n"
            "\n"
            "The history of f18 was rewritten to:\n"
            "\n"
            "* Put the code under /flang/.\n"
            "* Linearize the history.\n"
            "* Rewrite commit messages so that issue and PR numbers point to the old repository.\n"
            "\n"
            "Updates: flang-compiler/f18#876 (submission into llvm-project)\n"
            "Mailing-list: http://lists.llvm.org/pipermail/llvm-dev/2020-January/137989.html ([llvm-dev] Flang landing in the monorepo - next Monday!)\n"
            "Mailing-list: http://lists.llvm.org/pipermail/llvm-dev/2019-December/137661.html ([llvm-dev] Flang landing in the monorepo)\n";

        merge_llvm_project_tree(&new_tree, &new_tree, llvm_project_tree);

        const git_oid *parents[2] = {};
        parents[0] = &llvm_project_head;
        parents[1] = &new_commit_id;

        git_oid new_head_merged;
        check(git_commit_create_from_ids(
            &new_head_merged,
            repo,
            NULL,
            merge_commit_author,
            merge_commit_author,
            NULL,
            merge_message,
            &new_tree,
            2,
            parents
        ), "git_commit_create_from_ids", NULL);

        git_signature_free(merge_commit_author);

        git_tree_free(llvm_project_tree);

        git_reference *ref;
        check(git_reference_create(
                &ref,
                repo,
                "refs/heads/rewritten-history-v4-llvm-project-merge",
                &new_head_merged,
                1,
                "flatten.cpp update"
            ),
            "git_reference_create", NULL);
        git_reference_free(ref);

        git_oid_nfmt(buf, 12, &new_head_merged);
        printf("Done; rewritten-history-v4-llvm-project-merge => %s\n", buf);
    }
    printf("  ... all done\n");

    git_oid origin_master;
    git_reference_name_to_id(&origin_master, repo, "refs/remotes/origin/master");
    git_oid_nfmt(buf, 12, &origin_master);
    printf("Start point was origin/master => %s\n", buf);

    git_revwalk_free(walk);
    git_repository_free(repo);

    return 0;
}
