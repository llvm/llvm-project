//===-- ClangTidyPropertyGrid.cs - UI for configuring clang-tidy -*- C# -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class contains a UserControl consisting of a .NET PropertyGrid control
// allowing configuration of checks and check options for ClangTidy.
//
//===----------------------------------------------------------------------===//
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using Microsoft.VisualStudio.Shell;

namespace LLVM.ClangTidy
{
    /// <summary>
    ///  A UserControl displaying a PropertyGrid allowing configuration of clang-tidy
    ///  checks and check options, as well as serialization and deserialization of
    ///  clang-tidy configuration files.  When a configuration file is loaded, the
    ///  entire chain of configuration files is analyzed based on the file path,
    ///  and quick access is provided to edit or view any of the files in the
    ///  configuration chain, allowing easy visualization of where values come from
    ///  (similar in spirit to the -explain-config option of clang-tidy).
    /// </summary>
    public partial class ClangTidyPropertyGrid : UserControl
    {
        /// <summary>
        /// The sequence of .clang-tidy configuration files, starting from the root
        /// of the filesystem, down to the selected file.
        /// </summary>
        List<KeyValuePair<string, ClangTidyProperties>> PropertyChain_ = null;

        public ClangTidyPropertyGrid()
        {
            InitializeComponent();
            InitializeSettings();
        }

        private enum ShouldCancel
        {
            Yes,
            No,
        }

        public void SaveSettingsToStorage()
        {
            PersistUnsavedChanges(false);
        }

        private ShouldCancel PersistUnsavedChanges(bool PromptFirst)
        {
            var UnsavedResults = PropertyChain_.Where(x => x.Key != null && x.Value.GetHasUnsavedChanges());
            if (UnsavedResults.Count() == 0)
                return ShouldCancel.No;

            bool ShouldSave = false;
            if (PromptFirst)
            {
                var Response = MessageBox.Show(
                    "You have unsaved changes!  Do you want to save before loading a new file?",
                    "clang-tidy",
                    MessageBoxButtons.YesNoCancel);

                ShouldSave = (Response == DialogResult.Yes);
                if (Response == DialogResult.Cancel)
                    return ShouldCancel.Yes;
            }
            else
                ShouldSave = true;

            if (ShouldSave)
            {
                foreach (var Result in UnsavedResults)
                {
                    ClangTidyConfigParser.SerializeClangTidyFile(Result.Value, Result.Key);
                    Result.Value.SetHasUnsavedChanges(false);
                }
            }
            return ShouldCancel.No;
        }

        public void InitializeSettings()
        {
            PropertyChain_ = new List<KeyValuePair<string, ClangTidyProperties>>();
            PropertyChain_.Add(new KeyValuePair<string, ClangTidyProperties>(null, ClangTidyProperties.RootProperties));
            reloadPropertyChain();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            ShouldCancel Cancel = PersistUnsavedChanges(true);
            if (Cancel == ShouldCancel.Yes)
                return;

            using (OpenFileDialog D = new OpenFileDialog())
            {
                D.Filter = "Clang Tidy files|.clang-tidy";
                D.CheckPathExists = true;
                D.CheckFileExists = true;

                if (D.ShowDialog() == DialogResult.OK)
                {
                    PropertyChain_.Clear();
                    PropertyChain_ = ClangTidyConfigParser.ParseConfigurationChain(D.FileName);
                    textBox1.Text = D.FileName;
                    reloadPropertyChain();
                }
            }
        }

        private static readonly string DefaultText = "(Default)";
        private static readonly string BrowseText = "Browse for a file to edit its properties";

        /// <summary>
        /// After a new configuration file is chosen, analyzes the directory hierarchy
        /// and finds all .clang-tidy files in the path, parses them and updates the
        /// PropertyGrid and quick-access LinkLabel control to reflect the new property
        /// chain.
        /// </summary>
        private void reloadPropertyChain()
        {
            StringBuilder LinkBuilder = new StringBuilder();
            LinkBuilder.Append(DefaultText);
            LinkBuilder.Append(" > ");
            int PrefixLength = LinkBuilder.Length;

            if (PropertyChain_.Count == 1)
                LinkBuilder.Append(BrowseText);
            else
                LinkBuilder.Append(PropertyChain_[PropertyChain_.Count - 1].Key);

            linkLabelPath.Text = LinkBuilder.ToString();

            // Given a path like D:\Foo\Bar\Baz, construct a LinkLabel where individual
            // components of the path are clickable iff they contain a .clang-tidy file.
            // Clicking one of the links then updates the PropertyGrid to display the
            // selected .clang-tidy file.
            ClangTidyProperties LastProps = ClangTidyProperties.RootProperties;
            linkLabelPath.Links.Clear();
            linkLabelPath.Links.Add(0, DefaultText.Length, LastProps);
            foreach (var Prop in PropertyChain_.Skip(1))
            {
                LastProps = Prop.Value;
                string ClangTidyFolder = Path.GetFileName(Prop.Key);
                int ClangTidyFolderOffset = Prop.Key.Length - ClangTidyFolder.Length;
                linkLabelPath.Links.Add(PrefixLength + ClangTidyFolderOffset, ClangTidyFolder.Length, LastProps);
            }
            propertyGrid1.SelectedObject = LastProps;
        }

        private void propertyGrid1_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            ClangTidyProperties Props = (ClangTidyProperties)propertyGrid1.SelectedObject;
            Props.SetHasUnsavedChanges(true);

            // When a CategoryVerb is selected, perform the corresponding action.
            PropertyDescriptor Property = e.ChangedItem.PropertyDescriptor;
            if (!(e.ChangedItem.Value is CategoryVerb))
                return;

            CategoryVerb Action = (CategoryVerb)e.ChangedItem.Value;
            if (Action == CategoryVerb.None)
                return;

            var Category = Property.Attributes.OfType<CategoryAttribute>().FirstOrDefault();
            if (Category == null)
                return;
            var SameCategoryProps = Props.GetProperties(new Attribute[] { Category });
            foreach (PropertyDescriptor P in SameCategoryProps)
            {
                if (P == Property)
                    continue;
                switch (Action)
                {
                    case CategoryVerb.Disable:
                        P.SetValue(propertyGrid1.SelectedObject, false);
                        break;
                    case CategoryVerb.Enable:
                        P.SetValue(propertyGrid1.SelectedObject, true);
                        break;
                    case CategoryVerb.Inherit:
                        P.ResetValue(propertyGrid1.SelectedObject);
                        break;
                }
            }
            Property.ResetValue(propertyGrid1.SelectedObject);
            propertyGrid1.Invalidate();
        }

        private void linkLabelPath_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            ClangTidyProperties Props = (ClangTidyProperties)e.Link.LinkData;
            propertyGrid1.SelectedObject = Props;
        }
    }
}
